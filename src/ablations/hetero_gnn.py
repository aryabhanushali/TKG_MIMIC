"""Heterogeneous TKG-GNN with a real graph-message-passing layer, in the same
DeepHit competing-risks survival framing as tgn_survival.py.

Two variants, selected by the HETERO_VARIANT env var:

  full   (default) -- R-GCN message-passing over the ontology graph (ICD code
          -> category -> chapter; drug -> drug class; lab -> lab category)
          enriches each concept's embedding, THEN the same temporal
          Transformer sequence encoder as tgn_survival.py processes the
          patient's ordered event sequence. This tests: does injecting real
          graph structure into concept representations help, on top of the
          existing temporal-sequence model?

  static -- the same R-GCN-enriched concept embeddings, but NO temporal
          Transformer and NO time encoding at all: a masked mean-pool over a
          patient's (unordered) set of enriched-concept + edge-type + value
          embeddings. This isolates graph connectivity from time: if `static`
          performs close to `full`, graph structure (not event ordering/
          timestamps) is doing the work; if `static` performs much worse,
          temporal structure matters more than graph structure.

Requires tkg_output/ontology_edges.csv and node_index_v2.csv -- run
`python -u -m src.ablations.build_ontology` first.

All survival machinery (time bins, DeepHit loss, corrected competing-risks
horizon evaluation, MIN_EPOCHS floor) is imported from tgn_survival.py
unchanged, so results are directly comparable to Cox / XGBoost / the plain
TKG-Transformer -- same labels, same evaluation rule, same checkpoint-
selection discipline.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import RGCNConv

from src.config import OUTPUT_DIR, FIGURES_DIR, SEED, read_events_table
from src.tgn_model import (
    TimeEncoder, PatientEventsDataset, collate, _set_seed,
    MAX_SEQ_LEN, D_MODEL, N_HEADS, N_LAYERS, DROPOUT,
    BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, PATIENCE,
)
from src.tgn_survival import (
    CAUSES, CAUSE_TO_IDX, NUM_CAUSES, NUM_TIME_BINS, HORIZON_DAYS, MIN_EPOCHS,
    _make_time_bins, _discretize, _deephit_nll_per_sample,
    _prepare_survival_targets, _evaluate_survival, _per_cause_auroc_at_horizons,
)

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
N_RGCN_LAYERS = 1
RGCN_NUM_BASES = 4

HETERO_VARIANT = os.environ.get("HETERO_VARIANT", "full").lower()
assert HETERO_VARIANT in ("full", "static"), \
    f"HETERO_VARIANT must be 'full' or 'static', got {HETERO_VARIANT!r}"

_dirname = "hetero_gnn_survival" if HETERO_VARIANT == "full" else "static_gnn_survival"
MODEL_DIR = os.path.join(OUTPUT_DIR, _dirname if SEED == 42 else f"{_dirname}_seed{SEED}")


# --------------------------------------------------------------------------- #
# Shared: R-GCN concept enrichment over the ontology graph                    #
# --------------------------------------------------------------------------- #
class OntologyConceptTable(nn.Module):
    """Base concept embeddings + k R-GCN layers over the static ontology
    graph. A concept with no ontology parent (e.g. most drugs -- only 9.3%
    of drug concepts resolve to a class in the current dictionary) still
    gets R-GCN's root transformation applied, but receives no neighbor
    message, since it has no edges."""
    def __init__(self, n_concepts, n_relations, edge_index, edge_type,
                 d_model=D_MODEL, n_layers=N_RGCN_LAYERS,
                 num_bases=RGCN_NUM_BASES, dropout=DROPOUT):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)
        self.node_emb = nn.Embedding(n_concepts, d_model)
        nn.init.normal_(self.node_emb.weight, std=0.02)
        self.rgcn = nn.ModuleList([
            RGCNConv(d_model, d_model, num_relations=n_relations, num_bases=num_bases)
            for _ in range(n_layers)
        ])
        self.rgcn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self) -> torch.Tensor:
        x0 = self.node_emb.weight
        x = x0
        for layer in self.rgcn:
            x = layer(x, self.edge_index, self.edge_type)
            x = F.gelu(x)
            x = self.rgcn_dropout(x)
        return self.norm(x + x0)   # residual: keep leaf-specific signal


# --------------------------------------------------------------------------- #
# Variant 1: full graph + temporal Transformer (message-passing THEN order)   #
# --------------------------------------------------------------------------- #
class HeteroTKGFull(nn.Module):
    def __init__(self, n_concepts, n_relations, n_temporal_edge_types, n_static,
                 edge_index, edge_type, n_classes,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.concept_table = OntologyConceptTable(
            n_concepts, n_relations, edge_index, edge_type, d_model, dropout=dropout)
        self.edge_emb = nn.Embedding(n_temporal_edge_types, d_model)
        self.time_enc = TimeEncoder(d_model)
        self.value_proj = nn.Sequential(nn.Linear(2, d_model), nn.GELU())
        self.input_proj = nn.Linear(d_model * 4, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes))

    def forward(self, concept_idx, edge_type_idx, t, v_norm, v_present, static, mask):
        concept_table = self.concept_table()
        ce = concept_table[concept_idx]
        ee = self.edge_emb(edge_type_idx)
        te = self.time_enc(t)
        ve = self.value_proj(torch.stack([v_norm, v_present.float()], dim=-1))
        x = self.input_norm(self.input_proj(torch.cat([ce, ee, te, ve], dim=-1)))

        kpm = ~mask
        x = self.encoder(x, src_key_padding_mask=kpm)
        q = self.pool_query.expand(x.size(0), -1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=kpm)
        pooled = pooled.squeeze(1)

        s = self.static_proj(static)
        return self.head(torch.cat([pooled, s], dim=-1))


# --------------------------------------------------------------------------- #
# Variant 2: static graph only -- NO time encoding, NO sequence order         #
# --------------------------------------------------------------------------- #
class HeteroGNNStatic(nn.Module):
    """Graph-enriched concept embeddings, masked-mean pooled over a patient's
    UNORDERED set of events. No TimeEncoder term, no positional/sequence
    processing at all -- this model structurally cannot see event order or
    timing, only which concepts (and their relation types/values) occurred."""
    def __init__(self, n_concepts, n_relations, n_temporal_edge_types, n_static,
                 edge_index, edge_type, n_classes,
                 d_model=D_MODEL, dropout=DROPOUT):
        super().__init__()
        self.concept_table = OntologyConceptTable(
            n_concepts, n_relations, edge_index, edge_type, d_model, dropout=dropout)
        self.edge_emb = nn.Embedding(n_temporal_edge_types, d_model)
        self.value_proj = nn.Sequential(nn.Linear(2, d_model), nn.GELU())
        # 3 blocks (concept, edge_type, value) -- deliberately no time block.
        self.input_proj = nn.Linear(d_model * 3, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.event_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))

        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes))

    def forward(self, concept_idx, edge_type_idx, t, v_norm, v_present, static, mask):
        concept_table = self.concept_table()
        ce = concept_table[concept_idx]
        ee = self.edge_emb(edge_type_idx)
        ve = self.value_proj(torch.stack([v_norm, v_present.float()], dim=-1))
        x = self.input_norm(self.input_proj(torch.cat([ce, ee, ve], dim=-1)))
        x = self.event_mlp(x)

        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        s = self.static_proj(static)
        return self.head(torch.cat([pooled, s], dim=-1))


# --------------------------------------------------------------------------- #
# Survival wrapper (reshape flat logits -> (B, causes, time_bins))            #
# --------------------------------------------------------------------------- #
class HeteroSurvivalNet(nn.Module):
    def __init__(self, encoder_cls, n_concepts, n_relations, n_temporal_edge_types,
                 n_static, edge_index, edge_type,
                 num_causes=NUM_CAUSES, num_time_bins=NUM_TIME_BINS):
        super().__init__()
        self.encoder = encoder_cls(
            n_concepts=n_concepts, n_relations=n_relations,
            n_temporal_edge_types=n_temporal_edge_types, n_static=n_static,
            edge_index=edge_index, edge_type=edge_type,
            n_classes=num_causes * num_time_bins,
        )
        self.num_causes = num_causes
        self.num_time_bins = num_time_bins

    def forward(self, concept_idx, edge_type_idx, t, v_norm, v_present, static, mask):
        flat = self.encoder(concept_idx, edge_type_idx, t, v_norm, v_present, static, mask)
        return flat.view(-1, self.num_causes, self.num_time_bins)


# --------------------------------------------------------------------------- #
# Data prep (ontology graph + train-only concept restriction + survival y)    #
# --------------------------------------------------------------------------- #
def _prepare_hetero_survival_data():
    print("Loading modeling artifacts (v2 with ontology)...")
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    static = pd.read_csv(os.path.join(MODELING_DIR, "static_features.csv"))
    events = read_events_table()
    edge_types = pd.read_csv(os.path.join(MODELING_DIR, "edge_types.csv"))
    nodes_v2 = pd.read_csv(os.path.join(OUTPUT_DIR, "node_index_v2.csv"))
    ontology = pd.read_csv(os.path.join(OUTPUT_DIR, "ontology_edges.csv"))
    print(f"  events={len(events):,}, nodes_v2={len(nodes_v2):,}, "
          f"ontology={len(ontology):,}, temporal_edge_types={len(edge_types)}")

    concept_nodes = nodes_v2[nodes_v2["fact_type"] != "patient"].copy().reset_index(drop=True)
    concept_nodes["gnn_idx"] = np.arange(1, len(concept_nodes) + 1, dtype=np.int64)
    nidx_to_gnn = dict(zip(concept_nodes["node_idx"], concept_nodes["gnn_idx"]))

    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    train_concepts_global = set(
        events.loc[events["subject_id"].isin(train_ids), "concept_node_idx"].unique())

    events = events.copy()
    events["c_emb_idx"] = events["concept_node_idx"].map(nidx_to_gnn).astype("Int64")
    n_dropped = int(events["c_emb_idx"].isna().sum())
    if n_dropped:
        print(f"  WARN: dropping {n_dropped:,} events with unmapped concept")
        events = events.dropna(subset=["c_emb_idx"])
    events["c_emb_idx"] = events["c_emb_idx"].astype(np.int64)
    is_train_concept = events["concept_node_idx"].isin(train_concepts_global)
    n_oov = int((~is_train_concept).sum())
    events.loc[~is_train_concept, "c_emb_idx"] = 0
    print(f"  train-only concept restriction: {n_oov:,} events mapped to UNK "
          f"({n_oov / max(len(events), 1) * 100:.2f}%)")

    events = events.sort_values(["subject_id", "relative_days"]).reset_index(drop=True)
    if "value_norm" not in events.columns:
        events["value_norm"] = 0.0
    if "value_present" not in events.columns:
        events["value_present"] = 0.0
    events["value_norm"] = events["value_norm"].astype(np.float32)
    events["value_present"] = events["value_present"].astype(np.float32)

    events_by_sid: dict[int, tuple] = {}
    for sid, g in events.groupby("subject_id"):
        events_by_sid[int(sid)] = (
            g["c_emb_idx"].to_numpy(dtype=np.int64),
            g["edge_type_idx"].to_numpy(dtype=np.int64),
            g["relative_days"].to_numpy(dtype=np.float32),
            g["value_norm"].to_numpy(dtype=np.float32),
            g["value_present"].to_numpy(dtype=np.float32),
        )
    for sid in labels["subject_id"]:
        if int(sid) not in events_by_sid:
            # Single UNK/no-value placeholder, not a truly empty sequence --
            # see src.tgn_model._prepare_data for why an all-masked row would
            # otherwise NaN out of the Transformer's attention softmax.
            events_by_sid[int(sid)] = tuple(np.zeros(1, dtype=t) for t in
                                             (np.int64, np.int64, np.float32, np.float32, np.float32))

    static_cols = ["age_at_index", "cci_score", "num_cardiometa_conditions",
                   "had_icu_stay", "female"]
    static_train = static[static["subject_id"].isin(train_ids)][static_cols]
    mu = static_train.mean()
    sd = static_train.std().replace(0, 1.0)
    static_norm = (static[static_cols] - mu) / sd
    static_arr_by_sid = dict(zip(static["subject_id"].to_numpy(),
                                  static_norm.to_numpy(dtype=np.float32)))

    splits = {s: labels.loc[labels["split"] == s, "subject_id"].tolist()
              for s in ("train", "val", "test")}

    ontology = ontology.copy()
    ontology["src_gnn"] = ontology["src_node_idx"].map(nidx_to_gnn)
    ontology["dst_gnn"] = ontology["dst_node_idx"].map(nidx_to_gnn)
    ontology = ontology.dropna(subset=["src_gnn", "dst_gnn"]).copy()
    ontology["src_gnn"] = ontology["src_gnn"].astype(np.int64)
    ontology["dst_gnn"] = ontology["dst_gnn"].astype(np.int64)

    onto_edge_types = sorted(ontology["edge_type"].unique().tolist())
    etype_to_idx = {e: i for i, e in enumerate(onto_edge_types)}
    ontology["etype_idx"] = ontology["edge_type"].map(etype_to_idx).astype(np.int64)
    n_etypes = len(onto_edge_types)

    fwd_src = ontology["src_gnn"].to_numpy()
    fwd_dst = ontology["dst_gnn"].to_numpy()
    fwd_etype = ontology["etype_idx"].to_numpy()
    edge_index = np.stack([
        np.concatenate([fwd_src, fwd_dst]),
        np.concatenate([fwd_dst, fwd_src]),
    ])
    edge_type_arr = np.concatenate([fwd_etype, fwd_etype + n_etypes])
    edge_index_t = torch.from_numpy(edge_index).long()
    edge_type_t = torch.from_numpy(edge_type_arr).long()
    print(f"  ontology graph: nodes={len(concept_nodes):,}, "
          f"edges (with reverse)={edge_index_t.shape[1]:,}, relations={2 * n_etypes}")

    return {
        "events_by_sid": events_by_sid,
        "static_by_sid": static_arr_by_sid,
        "splits": splits,
        "n_concepts": len(concept_nodes) + 1,
        "n_relations": 2 * n_etypes,
        "n_temporal_edge_types": len(edge_types),
        "n_static": len(static_cols),
        "edge_index": edge_index_t,
        "edge_type": edge_type_t,
        "labels_df": labels,
    }


# --------------------------------------------------------------------------- #
# Train + evaluate (mirrors tgn_survival.train_and_eval, MIN_EPOCHS floor)    #
# --------------------------------------------------------------------------- #
def train_and_eval() -> None:
    _set_seed()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Variant: {HETERO_VARIANT}")
    d = _prepare_hetero_survival_data()
    labels_df = d["labels_df"]

    train_durations = labels_df.loc[
        labels_df["subject_id"].isin(d["splits"]["train"]), "time_to_event_days"
    ].to_numpy(dtype=np.float32)
    time_edges = _make_time_bins(train_durations, NUM_TIME_BINS)
    print(f"  time bin edges (days): {time_edges.round(1).tolist()}")

    survival_targets = _prepare_survival_targets(labels_df, time_edges)
    label_by_sid = {sid: int(t[1]) for sid, t in survival_targets.items()}
    duration_by_sid = {sid: int(t[0]) for sid, t in survival_targets.items()}

    def _make_loader(sids, shuffle):
        ds = PatientEventsDataset(sids, d["events_by_sid"], d["static_by_sid"],
                                   label_by_sid, max_len=MAX_SEQ_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                           collate_fn=collate, num_workers=0)

    train_loader = _make_loader(d["splits"]["train"], shuffle=True)
    val_loader = _make_loader(d["splits"]["val"], shuffle=False)
    test_loader = _make_loader(d["splits"]["test"], shuffle=False)

    # Verified working on MPS this session (RGCNConv forward+backward on a
    # small synthetic graph); if this errors on a different torch/PyG version,
    # fall back to CPU rather than debugging MPS scatter-op support live.
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  device: {device}")

    encoder_cls = HeteroTKGFull if HETERO_VARIANT == "full" else HeteroGNNStatic
    model = HeteroSurvivalNet(
        encoder_cls=encoder_cls,
        n_concepts=d["n_concepts"], n_relations=d["n_relations"],
        n_temporal_edge_types=d["n_temporal_edge_types"], n_static=d["n_static"],
        edge_index=d["edge_index"], edge_type=d["edge_type"],
        num_causes=NUM_CAUSES, num_time_bins=NUM_TIME_BINS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    counts = np.zeros(NUM_CAUSES + 1)
    for sid in d["splits"]["train"]:
        counts[label_by_sid[sid]] += 1
    weights = np.ones_like(counts)
    weights[1:] = (counts.sum() / (NUM_CAUSES * counts[1:].clip(min=1)))
    weights = weights / weights.mean()
    print(f"  event weights (0..K): {weights.round(3).tolist()}")
    sample_weight_by_event = torch.tensor(weights, dtype=torch.float32, device=device)

    def weighted_deephit_nll(logits, dur_idx, evt_idx):
        per_sample = _deephit_nll_per_sample(logits, dur_idx, evt_idx)
        w = sample_weight_by_event[evt_idx]
        return (per_sample * w).mean()

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_metric = -1.0
    best_epoch = -1
    no_improve = 0
    history = []
    best_state = None

    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE}, "
          f"min_epochs={MIN_EPOCHS})...")
    print("Selection metric: mean per-cause AUROC at 3-yr horizon on val set\n")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            for k in ("concept_idx", "edge_type_idx", "t",
                      "v_norm", "v_present", "static", "mask"):
                batch[k] = batch[k].to(device)
            sids = batch["sid"].tolist()
            evt_idx = torch.tensor([label_by_sid[s] for s in sids], dtype=torch.long, device=device)
            dur_idx = torch.tensor([duration_by_sid[s] for s in sids], dtype=torch.long, device=device)
            optim.zero_grad()
            logits = model(batch["concept_idx"], batch["edge_type_idx"], batch["t"],
                           batch["v_norm"], batch["v_present"], batch["static"], batch["mask"])
            loss = weighted_deephit_nll(logits, dur_idx, evt_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        train_loss /= max(n_batches, 1)

        cif_val, sids_val = _evaluate_survival(model, val_loader, device)
        val_metrics = _per_cause_auroc_at_horizons(cif_val, sids_val, labels_df, time_edges, HORIZON_DAYS)
        mean3y = float(val_metrics[val_metrics["horizon_days"] == 1095]["auroc"].mean(skipna=True))
        dt = time.time() - t0
        print(f"  ep {epoch:02d}  loss={train_loss:.4f}  val_mean_AUROC@3y={mean3y:.4f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_mean_auroc_3y": mean3y, "time_s": dt})
        if epoch < MIN_EPOCHS:
            continue
        if mean3y > best_metric:
            best_metric = mean3y; best_epoch = epoch; no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch}, "
                      f"val mean AUROC@3y={best_metric:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating on test set...")
    cif_test, sids_test = _evaluate_survival(model, test_loader, device)
    test_metrics = _per_cause_auroc_at_horizons(cif_test, sids_test, labels_df, time_edges, HORIZON_DAYS)
    print(f"\n=== HETERO-GNN ({HETERO_VARIANT}) TEST METRICS ===")
    pivot_auc = test_metrics.pivot(index="cause", columns="horizon_days", values="auroc").round(3)
    pivot_pr = test_metrics.pivot(index="cause", columns="horizon_days", values="auprc").round(3)
    print("\nAUROC:\n" + pivot_auc.to_string())
    print("\nAUPRC:\n" + pivot_pr.to_string())

    test_metrics.to_csv(os.path.join(MODEL_DIR, "test_metrics.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(MODEL_DIR, "history.csv"), index=False)
    torch.save({"state_dict": best_state, "best_epoch": best_epoch, "best_val_metric": best_metric},
               os.path.join(MODEL_DIR, "best_model.pt"))

    print(f"\nSaved:\n  {os.path.join(MODEL_DIR, 'test_metrics.csv')}\n"
          f"  {os.path.join(MODEL_DIR, 'history.csv')}\n"
          f"  {os.path.join(MODEL_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    train_and_eval()
