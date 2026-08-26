"""GNNExplainer instance explanations for TGN-Survival (torch_geometric).

The pool-attention weights used in `explain.py` say *where the model looks*,
but not *what it needs*: a high-attention event may be redundant, and a
low-attention one may still be necessary. This module runs PyG's official
`torch_geometric.explain.GNNExplainer` (Ying et al. 2019) on the trained
`TKGSurvivalNet`.

Each patient's pre-index event sequence is treated as a graph whose nodes are
the events. We wrap the survival model so that GNNExplainer's learned node mask
acts as a soft gate g in [0,1] on the per-event embeddings: the explainer
optimizes g to preserve the model's own predicted competing-risks class while
penalizing mask size/entropy, yielding the *minimal sufficient* event set the
model relies on. Node features are ones(N,1), so the returned `node_mask` is
exactly the per-event necessity score.

Inputs : tkg_output/tgn_survival/best_model.pt  (+ modeling artifacts)
Outputs: tkg_output/explain/gnn_per_event_importance.csv
         tkg_output/explain/gnn_concept_importance_by_cause.csv
         tkg_output/explain/gnn_explainer_fidelity.csv
         tkg_output/figures/fig17_gnn_explainer_concepts.png
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from src.config import OUTPUT_DIR, FIGURES_DIR
from src.tgn_model import _prepare_data, _set_seed, MAX_SEQ_LEN
from src.tgn_survival import TKGSurvivalNet, CAUSES, NUM_CAUSES, NUM_TIME_BINS, MODEL_DIR
from src.explain import _load_concept_remap_lookup
from src.fidelity_stats import summarize_fidelity, print_fidelity_summary

EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explain")

MAX_PATIENTS_PER_CAUSE = 30   # true-positive patients explained per cause
GNN_EPOCHS = 100              # GNNExplainer mask-optimization epochs/patient
GNN_LR = 0.01
KEEP_FRAC = 0.20              # fidelity: keep this top fraction of events
TOP_N_CONCEPTS = 15
EPS = 1e-9

CAUSE_COLORS = {"MI": "#d62728", "Stroke": "#9467bd", "HF": "#ff7f0e",
                "AF": "#1f77b4", "PAD": "#2ca02c"}


class _PatientGraphModel(nn.Module):
    """Wrap TKGSurvivalNet so it looks like a PyG graph classifier.

    forward(x, edge_index, **event_kwargs) -> (1, C*T) class probabilities.
    ``x`` is (N, 1); GNNExplainer feeds in ``x * sigmoid(node_mask)``, which we
    read as a per-event gate applied to the event embeddings. ``edge_index`` is
    ignored (the model has no message passing over events; the temporal
    structure lives in the time-encoding and self-attention).
    """

    def __init__(self, surv: TKGSurvivalNet):
        super().__init__()
        self.surv = surv

    def forward(self, x, edge_index, *, concept_idx, edge_type_idx, t,
                v_norm, v_present, static):
        enc = self.surv.encoder
        gate = x.squeeze(-1).unsqueeze(0)                 # (1, N)
        ci = concept_idx.unsqueeze(0)
        ce = enc.concept_emb(ci)
        ee = enc.edge_emb(edge_type_idx.unsqueeze(0))
        te = enc.time_enc(t.unsqueeze(0))
        ve = enc.value_proj(torch.stack(
            [v_norm.unsqueeze(0), v_present.unsqueeze(0).float()], dim=-1))
        h = enc.input_proj(torch.cat([ce, ee, te, ve], dim=-1))
        h = enc.input_norm(h)
        h = h * gate.unsqueeze(-1)                        # soft event gate

        h = enc.encoder(h)                                # all events valid
        q = enc.pool_query.expand(h.size(0), -1, -1)
        pooled, _ = enc.pool_attn(q, h, h)
        pooled = pooled.squeeze(1)
        s = enc.static_proj(static.unsqueeze(0) if static.dim() == 1 else static)
        out = enc.head(torch.cat([pooled, s], dim=-1))    # (1, C*T)
        return F.softmax(out, dim=-1)


def _patient_tensors(events_by_sid, static_by_sid, sid, device):
    """Per-event tensors for one patient (truncated to MAX_SEQ_LEN), on device."""
    c, e, t, vn, vp = events_by_sid[sid]
    if len(c) > MAX_SEQ_LEN:
        c, e, t, vn, vp = (a[-MAX_SEQ_LEN:] for a in (c, e, t, vn, vp))
    n = len(c)
    return {
        "concept_idx": torch.as_tensor(c, dtype=torch.long, device=device),
        "edge_type_idx": torch.as_tensor(e, dtype=torch.long, device=device),
        "t": torch.as_tensor(t, dtype=torch.float, device=device),
        "v_norm": torch.as_tensor(vn, dtype=torch.float, device=device),
        "v_present": torch.as_tensor(vp, dtype=torch.float, device=device),
        "static": torch.as_tensor(static_by_sid[sid], dtype=torch.float,
                                  device=device),
    }, n


def _cif_probs(wrapped, x, kwargs):
    """Class-probability vector (C*T,) for a gate of all ones."""
    with torch.no_grad():
        p = wrapped(x, None, **kwargs).squeeze(0)
    return p


def _fidelity_row(wrapped, x, kwargs, node_imp, sid, n, keep_frac):
    """Count-matched faithfulness check (top-k vs random-k at the same budget):

      sufficiency      keep only the k events; KL(orig || keep). The top-k
                       (by learned mask) should give a *lower* KL than random-k.
      comprehensiveness drop the k events; KL(orig || drop). Dropping the top-k
                       should give a *higher* KL than dropping random-k.
    """
    p_orig = _cif_probs(wrapped, x, kwargs)
    log_orig = torch.log(p_orig + EPS)

    def _kl(idx, base):
        xx = torch.full_like(x, base); xx[idx] = 1.0 - base
        p = _cif_probs(wrapped, xx, kwargs)
        return float((p_orig * (log_orig - torch.log(p + EPS))).sum())

    k = max(int(round(n * keep_frac)), 1)
    top = np.argsort(-node_imp)[:k]
    rand = np.random.default_rng(int(sid)).choice(n, size=k, replace=False)
    return {"subject_id": int(sid), "n_events": int(n), "keep_frac": keep_frac,
            "kl_keep_top": _kl(top, 0.0), "kl_keep_random": _kl(rand, 0.0),
            "kl_drop_top": _kl(top, 1.0), "kl_drop_random": _kl(rand, 1.0),
            "mean_mask": float(node_imp.mean())}


def run() -> None:
    _set_seed()
    os.makedirs(EXPLAIN_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data + model...")
    (events_by_sid, static_by_sid, label_by_sid,
     splits, n_concepts, n_edge_types, n_static, labels_df) = _prepare_data()
    emb_to_concept, emb_to_facttype = _load_concept_remap_lookup(labels_df)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    surv = TKGSurvivalNet(n_concepts=n_concepts, n_edge_types=n_edge_types,
                          n_static=n_static).to(device)
    ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                      map_location=device, weights_only=False)
    surv.load_state_dict(ckpt["state_dict"])
    surv.eval()
    print(f"  loaded best_model.pt (epoch {ckpt.get('best_epoch', '?')})  "
          f"device={device}")

    wrapped = _PatientGraphModel(surv).to(device).eval()

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=GNN_EPOCHS, lr=GNN_LR),
        explanation_type="model",
        node_mask_type="object",        # one scalar necessity mask per event
        edge_mask_type=None,            # no message passing over events
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="probs",
        ),
    )

    sid_to_label = dict(zip(labels_df["subject_id"], labels_df["endpoint_type"]))
    selected = []
    for cause in CAUSES:
        c_sids = [s for s in splits["test"]
                  if sid_to_label.get(s) == cause and len(events_by_sid.get(s, [[]])[0]) > 0]
        selected.extend(c_sids[:MAX_PATIENTS_PER_CAUSE])
    print(f"  explaining {len(selected):,} true-positive test patients "
          f"(<= {MAX_PATIENTS_PER_CAUSE} per cause) with PyG GNNExplainer "
          f"({GNN_EPOCHS} epochs each)")

    rows, fidelity_rows = [], []
    for n_done, sid in enumerate(selected, 1):
        kwargs, n = _patient_tensors(events_by_sid, static_by_sid, sid, device)
        x = torch.ones(n, 1, device=device)
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

        explanation = explainer(x, edge_index, **kwargs)
        node_imp = explanation.node_mask.detach().cpu().numpy().reshape(-1)

        fidelity_rows.append(
            _fidelity_row(wrapped, x, kwargs, node_imp, sid, n, KEEP_FRAC))
        c = kwargs["concept_idx"].cpu().numpy()
        e = kwargs["edge_type_idx"].cpu().numpy()
        t = kwargs["t"].cpu().numpy()
        ep = sid_to_label.get(int(sid), "censored")
        for j in range(n):
            rows.append({
                "subject_id": int(sid), "endpoint": ep,
                "concept_emb_idx": int(c[j]), "edge_type_idx": int(e[j]),
                "relative_days": float(t[j]),
                "mask_importance": float(node_imp[j]),
            })
        if n_done % 25 == 0 or n_done == len(selected):
            print(f"  explained {n_done}/{len(selected)} patients")

    per_event = pd.DataFrame(rows)
    per_event["concept_id"] = per_event["concept_emb_idx"].map(emb_to_concept)
    per_event["fact_type"] = per_event["concept_emb_idx"].map(emb_to_facttype)
    per_event_path = os.path.join(EXPLAIN_DIR, "gnn_per_event_importance.csv")
    per_event.to_csv(per_event_path, index=False)

    by_cause = (per_event.groupby(["endpoint", "concept_id", "fact_type"])
                .agg(mean_mask=("mask_importance", "mean"),
                     sum_mask=("mask_importance", "sum"),
                     n_appearances=("mask_importance", "size"),
                     n_patients=("subject_id", "nunique"))
                .reset_index())
    bc_path = os.path.join(EXPLAIN_DIR, "gnn_concept_importance_by_cause.csv")
    (by_cause.sort_values(["endpoint", "sum_mask"], ascending=[True, False])
     .to_csv(bc_path, index=False))

    fidelity = pd.DataFrame(fidelity_rows)
    fid_path = os.path.join(EXPLAIN_DIR, "gnn_explainer_fidelity.csv")
    fidelity.to_csv(fid_path, index=False)
    pct = int(KEEP_FRAC * 100)
    print("\nFidelity (mean over explained patients, top-k vs random-k):")
    print(f"  sufficiency  KL(keep top-{pct}%)  = {fidelity['kl_keep_top'].mean():.4f}"
          f"  vs random = {fidelity['kl_keep_random'].mean():.4f}  "
          "(top should be LOWER)")
    print(f"  comprehens.  KL(drop top-{pct}%)  = {fidelity['kl_drop_top'].mean():.4f}"
          f"  vs random = {fidelity['kl_drop_random'].mean():.4f}  "
          "(top should be HIGHER)")
    print(f"  mean learned mask density        = {fidelity['mean_mask'].mean():.3f}")

    # Per-patient paired significance testing -- a mean alone can be driven by
    # a handful of outlier patients; report win-rate + paired-t + Wilcoxon so
    # "the important set beats random" is a claim about a typical patient,
    # not just the average one. See src/fidelity_stats.py.
    fidelity_summary = summarize_fidelity(fidelity)
    fidelity_summary.to_csv(os.path.join(EXPLAIN_DIR, "gnn_explainer_fidelity_stats.csv"))
    print_fidelity_summary(fidelity_summary, "Plain TKG-Transformer")

    fig, axes = plt.subplots(1, 5, figsize=(22, 8))
    for ax, cause in zip(axes, CAUSES):
        sub = (by_cause[(by_cause["endpoint"] == cause) & (by_cause["n_patients"] >= 3)]
               .sort_values("sum_mask", ascending=False).head(TOP_N_CONCEPTS).iloc[::-1])
        if sub.empty:
            ax.set_title(cause); ax.axis("off"); continue
        ax.barh(sub["concept_id"], sub["sum_mask"], color=CAUSE_COLORS[cause])
        ax.set_title(f"{cause}\n(top-{TOP_N_CONCEPTS} by necessity)",
                     fontweight="bold")
        ax.set_xlabel("Σ GNNExplainer node mask")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Minimal-sufficient TKG concepts per cause "
                 "(PyG GNNExplainer necessity mask on TGN-Survival)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(FIGURES_DIR, "fig17_gnn_explainer_concepts.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print(f"  {per_event_path}")
    print(f"  {bc_path}")
    print(f"  {fid_path}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    run()
