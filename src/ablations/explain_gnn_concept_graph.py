"""Item 6a: extend the GNNExplainer fidelity check (src/explain_gnn.py) to the
failed concept-graph model (attempt 1, "full" variant), to test whether the
checkpoint-selection/fidelity finding generalizes beyond the plain
TKG-Transformer, or was specific to that one architecture.

Structurally identical to explain_gnn.py -- the concept-graph model is still
a per-patient SEQUENCE model, just with R-GCN-enriched concept embeddings
instead of a plain lookup table, so the same "mask events as PyG graph
nodes" trick applies directly. The only change is swapping the encoder's
concept-embedding lookup for its concept_table() R-GCN forward pass.

Output: tkg_output/explain/gnn_explainer_fidelity_concept_graph.csv
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from src.config import OUTPUT_DIR
from src.tgn_model import _set_seed, MAX_SEQ_LEN
from src.tgn_survival import CAUSES, NUM_CAUSES, NUM_TIME_BINS
from src.explain_gnn import _patient_tensors, _cif_probs, _fidelity_row, KEEP_FRAC, GNN_EPOCHS, GNN_LR, MAX_PATIENTS_PER_CAUSE
from src.ablations.hetero_gnn import HeteroSurvivalNet, HeteroTKGFull, _prepare_hetero_survival_data
from src.fidelity_stats import summarize_fidelity, print_fidelity_summary

EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explain")
MODEL_DIR = os.path.join(OUTPUT_DIR, "hetero_gnn_survival")


class _ConceptGraphPatientModel(nn.Module):
    """Same event-masking trick as explain_gnn.py's _PatientGraphModel, but
    for the concept-graph encoder: the R-GCN pass over the (fixed, shared)
    ontology graph is computed once and reused for every patient, since it
    doesn't depend on which patient is being explained."""

    def __init__(self, surv: HeteroSurvivalNet):
        super().__init__()
        self.surv = surv
        with torch.no_grad():
            self._concept_table = surv.encoder.concept_table()

    def forward(self, x, edge_index, *, concept_idx, edge_type_idx, t,
                v_norm, v_present, static):
        enc = self.surv.encoder
        gate = x.squeeze(-1).unsqueeze(0)
        ci = concept_idx.unsqueeze(0)
        ce = self._concept_table[ci]
        ee = enc.edge_emb(edge_type_idx.unsqueeze(0))
        te = enc.time_enc(t.unsqueeze(0))
        ve = enc.value_proj(torch.stack(
            [v_norm.unsqueeze(0), v_present.unsqueeze(0).float()], dim=-1))
        h = enc.input_proj(torch.cat([ce, ee, te, ve], dim=-1))
        h = enc.input_norm(h)
        h = h * gate.unsqueeze(-1)

        h = enc.encoder(h)
        q = enc.pool_query.expand(h.size(0), -1, -1)
        pooled, _ = enc.pool_attn(q, h, h)
        pooled = pooled.squeeze(1)
        s = enc.static_proj(static.unsqueeze(0) if static.dim() == 1 else static)
        out = enc.head(torch.cat([pooled, s], dim=-1))
        return F.softmax(out, dim=-1)


def run() -> None:
    _set_seed()
    os.makedirs(EXPLAIN_DIR, exist_ok=True)

    print("Loading concept-graph data + model...")
    d = _prepare_hetero_survival_data()
    events_by_sid, static_by_sid, splits, labels_df = (
        d["events_by_sid"], d["static_by_sid"], d["splits"], d["labels_df"])

    device = torch.device("cpu")  # small inference-only workload; avoids MPS/RGCN scatter risk
    surv = HeteroSurvivalNet(
        encoder_cls=HeteroTKGFull, n_concepts=d["n_concepts"], n_relations=d["n_relations"],
        n_temporal_edge_types=d["n_temporal_edge_types"], n_static=d["n_static"],
        edge_index=d["edge_index"], edge_type=d["edge_type"],
        num_causes=NUM_CAUSES, num_time_bins=NUM_TIME_BINS,
    ).to(device)
    ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=device, weights_only=False)
    surv.load_state_dict(ckpt["state_dict"])
    surv.eval()
    print(f"  loaded best_model.pt (epoch {ckpt.get('best_epoch', '?')})  device={device}")

    wrapped = _ConceptGraphPatientModel(surv).to(device).eval()

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=GNN_EPOCHS, lr=GNN_LR),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type=None,
        model_config=ModelConfig(mode="multiclass_classification", task_level="graph", return_type="probs"),
    )

    sid_to_label = dict(zip(labels_df["subject_id"], labels_df["endpoint_type"]))
    selected = []
    for cause in CAUSES:
        c_sids = [s for s in splits["test"]
                  if sid_to_label.get(s) == cause and len(events_by_sid.get(s, [[]])[0]) > 0]
        selected.extend(c_sids[:MAX_PATIENTS_PER_CAUSE])
    print(f"  explaining {len(selected):,} true-positive test patients "
          f"(<= {MAX_PATIENTS_PER_CAUSE} per cause)")

    fidelity_rows = []
    for n_done, sid in enumerate(selected, 1):
        kwargs, n = _patient_tensors(events_by_sid, static_by_sid, sid, device)
        x = torch.ones(n, 1, device=device)
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        explanation = explainer(x, edge_index, **kwargs)
        node_imp = explanation.node_mask.detach().cpu().numpy().reshape(-1)
        fidelity_rows.append(_fidelity_row(wrapped, x, kwargs, node_imp, sid, n, KEEP_FRAC))
        if n_done % 25 == 0 or n_done == len(selected):
            print(f"  explained {n_done}/{len(selected)} patients")

    fidelity = pd.DataFrame(fidelity_rows)
    fid_path = os.path.join(EXPLAIN_DIR, "gnn_explainer_fidelity_concept_graph.csv")
    fidelity.to_csv(fid_path, index=False)
    pct = int(KEEP_FRAC * 100)
    print("\nFidelity (concept-graph model, mean over explained patients, top-k vs random-k):")
    print(f"  sufficiency  KL(keep top-{pct}%)  = {fidelity['kl_keep_top'].mean():.4f}"
          f"  vs random = {fidelity['kl_keep_random'].mean():.4f}  (top should be LOWER)")
    print(f"  comprehens.  KL(drop top-{pct}%)  = {fidelity['kl_drop_top'].mean():.4f}"
          f"  vs random = {fidelity['kl_drop_random'].mean():.4f}  (top should be HIGHER)")

    fidelity_summary = summarize_fidelity(fidelity)
    fidelity_summary.to_csv(os.path.join(EXPLAIN_DIR, "gnn_explainer_fidelity_concept_graph_stats.csv"))
    print_fidelity_summary(fidelity_summary, "Concept-graph model (attempt 1)")

    print(f"\nSaved: {fid_path}")


if __name__ == "__main__":
    run()
