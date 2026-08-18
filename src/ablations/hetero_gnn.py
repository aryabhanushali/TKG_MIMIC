"""Heterogeneous TKG-GNN: R-GCN over ontology edges + temporal attention.

Architecture:
  1. Base concept embedding table  (n_concepts x d)
  2. R-GCN (k layers) propagates over ontology isA edges -> enriched concept emb.
  3. For each patient: per-event embedding = concept_emb + edge_type_emb + time_enc
  4. Transformer encoder over event sequence + attention-pool -> patient emb
  5. Concat with static features (age/CCI/...) -> MLP -> 6-class logits
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from torch_geometric.nn import RGCNConv

from src.config import OUTPUT_DIR, FIGURES_DIR, read_events_table
from src.tgn_model import (
    TimeEncoder, PatientEventsDataset, collate,
    ENDPOINT_ORDER, EP_TO_IDX,
    MAX_SEQ_LEN, D_MODEL, N_HEADS, N_LAYERS, DROPOUT,
    BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, PATIENCE,
    _per_endpoint, _overall, _set_seed,
)

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
HETERO_DIR = os.path.join(OUTPUT_DIR, "hetero_gnn")

N_RGCN_LAYERS = 1
RGCN_NUM_BASES = 4


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class HeteroTKG(nn.Module):
    def __init__(self, n_concepts: int, n_relations: int,
                 n_temporal_edge_types: int, n_static: int,
                 edge_index: torch.Tensor, edge_type: torch.Tensor,
                 n_classes: int = 6, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                 n_rgcn_layers: int = N_RGCN_LAYERS,
                 num_bases: int = RGCN_NUM_BASES,
                 dropout: float = DROPOUT):
        super().__init__()
        self.d_model = d_model

        # Static ontology graph (registered as buffers so they move with .to(device))
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

        # Concept (leaf + hub) embedding table
        self.node_emb = nn.Embedding(n_concepts, d_model)
        nn.init.normal_(self.node_emb.weight, std=0.02)

        # R-GCN stack over ontology
        self.rgcn = nn.ModuleList([
            RGCNConv(d_model, d_model, num_relations=n_relations,
                     num_bases=num_bases)
            for _ in range(n_rgcn_layers)
        ])
        self.rgcn_dropout = nn.Dropout(dropout)
        self.concept_norm = nn.LayerNorm(d_model)

        # Temporal patient encoder (same as TGN)
        self.edge_emb = nn.Embedding(n_temporal_edge_types, d_model)
        self.time_enc = TimeEncoder(d_model)
        self.value_proj = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(),
        )
        self.input_proj = nn.Linear(d_model * 4, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )

        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def compute_concept_table(self) -> torch.Tensor:
        x0 = self.node_emb.weight
        x = x0
        for layer in self.rgcn:
            x = layer(x, self.edge_index, self.edge_type)
            x = F.gelu(x)
            x = self.rgcn_dropout(x)
        # Residual: keep leaf-specific signal, ontology layer is additive
        return self.concept_norm(x + x0)

    def forward(self, concept_idx, edge_type_idx, t, v_norm, v_present,
                static, mask):
        concept_table = self.compute_concept_table()
        ce = concept_table[concept_idx]                  # (B, L, d)
        ee = self.edge_emb(edge_type_idx)
        te = self.time_enc(t)
        ve = self.value_proj(torch.stack([v_norm, v_present.float()], dim=-1))
        x = self.input_proj(torch.cat([ce, ee, te, ve], dim=-1))
        x = self.input_norm(x)

        kpm = ~mask
        x = self.encoder(x, src_key_padding_mask=kpm)
        q = self.pool_query.expand(x.size(0), -1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=kpm)
        pooled = pooled.squeeze(1)

        s = self.static_proj(static)
        return self.head(torch.cat([pooled, s], dim=-1))


# --------------------------------------------------------------------------- #
# Data prep                                                                   #
# --------------------------------------------------------------------------- #
def _prepare_hetero_data():
    print("Loading modeling artifacts (v2 with ontology)...")
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    static = pd.read_csv(os.path.join(MODELING_DIR, "static_features.csv"))
    events = read_events_table()
    edge_types = pd.read_csv(os.path.join(MODELING_DIR, "edge_types.csv"))
    nodes_v2 = pd.read_csv(os.path.join(OUTPUT_DIR, "node_index_v2.csv"))
    ontology = pd.read_csv(os.path.join(OUTPUT_DIR, "ontology_edges.csv"))
    print(f"  events={len(events):,}, nodes_v2={len(nodes_v2):,}, "
          f"ontology={len(ontology):,}, temporal_edge_types={len(edge_types)}")

    # Concept-only sub-table (we don't put patient nodes into the GNN graph)
    concept_nodes = nodes_v2[nodes_v2["fact_type"] != "patient"].copy()
    concept_nodes = concept_nodes.reset_index(drop=True)
    # idx 0 reserved for UNK token; concepts start at idx 1
    concept_nodes["gnn_idx"] = np.arange(1, len(concept_nodes) + 1, dtype=np.int64)
    nidx_to_gnn = dict(zip(concept_nodes["node_idx"], concept_nodes["gnn_idx"]))

    # Train-only concept restriction: events whose concept was never seen
    # in the training set get remapped to UNK (idx 0). This prevents the
    # model from quietly memorizing test-only concepts during embedding lookup.
    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    train_concepts_global = set(
        events.loc[events["subject_id"].isin(train_ids), "concept_node_idx"].unique()
    )
    events = events.copy()
    events["c_emb_idx"] = events["concept_node_idx"].map(nidx_to_gnn).astype("Int64")
    n_dropped = int(events["c_emb_idx"].isna().sum())
    if n_dropped:
        print(f"  WARN: dropping {n_dropped:,} events with unmapped concept")
        events = events.dropna(subset=["c_emb_idx"])
    events["c_emb_idx"] = events["c_emb_idx"].astype(np.int64)
    # Mask non-train concepts -> UNK (0)
    is_train_concept = events["concept_node_idx"].isin(train_concepts_global)
    n_oov = int((~is_train_concept).sum())
    events.loc[~is_train_concept, "c_emb_idx"] = 0
    print(f"  train-only concept restriction: {n_oov:,} events mapped to UNK "
          f"({n_oov / max(len(events), 1) * 100:.2f}%)")

    # Sort
    events = events.sort_values(["subject_id", "relative_days"]).reset_index(drop=True)

    # Default value columns if missing (back-compat with old events.csv)
    if "value_norm" not in events.columns:
        events["value_norm"] = 0.0
    if "value_present" not in events.columns:
        events["value_present"] = 0.0
    events["value_norm"] = events["value_norm"].astype(np.float32)
    events["value_present"] = events["value_present"].astype(np.float32)

    # Build per-patient event arrays (5-tuple with value features)
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
            events_by_sid[int(sid)] = (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

    # Static features: normalize with train stats
    static_cols = ["age_at_index", "cci_score", "num_cardiometa_conditions",
                   "had_icu_stay", "female"]
    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    static_train = static[static["subject_id"].isin(train_ids)][static_cols]
    mu = static_train.mean()
    sd = static_train.std().replace(0, 1.0)
    static_norm = (static[static_cols] - mu) / sd
    static_arr_by_sid = dict(zip(static["subject_id"].to_numpy(),
                                  static_norm.to_numpy(dtype=np.float32)))
    label_by_sid = dict(zip(labels["subject_id"],
                              labels["endpoint_type"].map(EP_TO_IDX)))

    splits = {s: labels.loc[labels["split"] == s, "subject_id"].tolist()
              for s in ("train", "val", "test")}

    # Build ontology edge_index + edge_type with bidirectional edges
    ontology["src_gnn"] = ontology["src_node_idx"].map(nidx_to_gnn)
    ontology["dst_gnn"] = ontology["dst_node_idx"].map(nidx_to_gnn)
    ontology = ontology.dropna(subset=["src_gnn", "dst_gnn"]).copy()
    ontology["src_gnn"] = ontology["src_gnn"].astype(np.int64)
    ontology["dst_gnn"] = ontology["dst_gnn"].astype(np.int64)

    onto_edge_types = sorted(ontology["edge_type"].unique().tolist())
    etype_to_idx = {e: i for i, e in enumerate(onto_edge_types)}
    ontology["etype_idx"] = ontology["edge_type"].map(etype_to_idx).astype(np.int64)
    n_etypes = len(onto_edge_types)
    print(f"  ontology edge types (x2 for reverse): {n_etypes} -> "
          f"{2 * n_etypes} relations")

    # Forward (child -> parent) + reverse (parent -> child)
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
          f"edges (with reverse)={edge_index_t.shape[1]:,}")

    return {
        "events_by_sid": events_by_sid,
        "static_by_sid": static_arr_by_sid,
        "label_by_sid": label_by_sid,
        "splits": splits,
        # +1 for UNK at idx 0 (OOV concepts and non-train concepts route here)
        "n_concepts": len(concept_nodes) + 1,
        "n_relations": 2 * n_etypes,
        "n_temporal_edge_types": len(edge_types),
        "n_static": len(static_cols),
        "edge_index": edge_index_t,
        "edge_type": edge_type_t,
        "labels_df": labels,
    }


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def _evaluate(model, loader, device):
    model.eval()
    logits_all, y_all, sid_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            for k in ("concept_idx", "edge_type_idx", "t",
                     "v_norm", "v_present", "static", "mask"):
                batch[k] = batch[k].to(device)
            logits = model(batch["concept_idx"], batch["edge_type_idx"],
                            batch["t"], batch["v_norm"], batch["v_present"],
                            batch["static"], batch["mask"])
            logits_all.append(logits.cpu())
            y_all.append(batch["label"])
            sid_all.append(batch["sid"])
    logits = torch.cat(logits_all)
    proba = F.softmax(logits, dim=-1).numpy()
    return proba, torch.cat(y_all).numpy(), torch.cat(sid_all).numpy()


def _plot_roc_compare(yte, hetero_proba, out_path):
    """4-model ROC: logreg, xgboost, tgn, hetero-gnn."""
    base = pd.read_csv(os.path.join(OUTPUT_DIR, "baseline", "predictions_test.csv"))
    tgn  = pd.read_csv(os.path.join(OUTPUT_DIR, "tgn", "predictions_test.csv"))
    base = base.set_index("subject_id")
    tgn  = tgn.set_index("subject_id")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for k, ep in enumerate(ENDPOINT_ORDER):
        ax = axes[k]
        j = EP_TO_IDX[ep]
        y_bin = (yte == j).astype(int)
        if y_bin.sum() == 0:
            ax.set_title(f"{ep} (no positives)")
            continue
        sources = [
            ("logreg", base[f"logreg_p_{ep}"].to_numpy(), 1.0, "C0"),
            ("xgboost", base[f"xgb_p_{ep}"].to_numpy(), 1.0, "C1"),
            ("TGN", tgn[f"tgn_p_{ep}"].to_numpy(), 1.5, "C2"),
            ("HeteroGNN", hetero_proba[:, j], 2.5, "C3"),
        ]
        for name, proba, lw, color in sources:
            fpr, tpr, _ = roc_curve(y_bin, proba)
            try:
                auc = roc_auc_score(y_bin, proba)
            except ValueError:
                auc = float("nan")
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                    linewidth=lw, color=color)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5)
        ax.set_title(f"{ep} (n={int(y_bin.sum())})", fontweight="bold")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Test ROC: baselines vs TGN vs Hetero-GNN",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #
def train_and_eval() -> None:
    _set_seed()
    os.makedirs(HETERO_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    d = _prepare_hetero_data()

    def _make_loader(sids, shuffle):
        ds = PatientEventsDataset(sids, d["events_by_sid"], d["static_by_sid"],
                                    d["label_by_sid"], max_len=MAX_SEQ_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                            collate_fn=collate, num_workers=0)

    train_loader = _make_loader(d["splits"]["train"], shuffle=True)
    val_loader = _make_loader(d["splits"]["val"], shuffle=False)
    test_loader = _make_loader(d["splits"]["test"], shuffle=False)

    # R-GCN on MPS can be flaky for scatter; fall back to CPU if needed
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  device: {device}")

    model = HeteroTKG(
        n_concepts=d["n_concepts"],
        n_relations=d["n_relations"],
        n_temporal_edge_types=d["n_temporal_edge_types"],
        n_static=d["n_static"],
        edge_index=d["edge_index"],
        edge_type=d["edge_type"],
        n_classes=len(ENDPOINT_ORDER),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    # Class-balanced cross-entropy
    train_labels = np.array([d["label_by_sid"][s] for s in d["splits"]["train"]])
    counts = np.bincount(train_labels, minlength=len(ENDPOINT_ORDER)).astype(float)
    inv = (1.0 / counts) * counts.sum() / len(ENDPOINT_ORDER)
    cw = torch.tensor(inv, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    optim = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_val_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    history = []
    best_state = None

    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE})...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            for k in ("concept_idx", "edge_type_idx", "t",
                     "v_norm", "v_present", "static", "mask"):
                batch[k] = batch[k].to(device)
            batch["label"] = batch["label"].to(device)
            optim.zero_grad()
            logits = model(batch["concept_idx"], batch["edge_type_idx"],
                            batch["t"], batch["v_norm"], batch["v_present"],
                            batch["static"], batch["mask"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        train_loss /= max(n_batches, 1)

        val_proba, val_y, _ = _evaluate(model, val_loader, device)
        val_macro = f1_score(val_y, val_proba.argmax(axis=1), average="macro")
        val_acc = accuracy_score(val_y, val_proba.argmax(axis=1))
        val_per_ep = _per_endpoint(val_y, val_proba)
        val_mean_auc = val_per_ep[val_per_ep["endpoint"] != "censored"]["auroc"].mean()
        dt = time.time() - t0
        print(f"  ep {epoch:02d}  loss={train_loss:.4f}  "
              f"val_macroF1={val_macro:.4f}  val_acc={val_acc:.4f}  "
              f"val_AUROC(mean)={val_mean_auc:.4f}  ({dt:.1f}s)")
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_macro_f1": val_macro, "val_acc": val_acc,
            "val_mean_auroc": val_mean_auc, "time_s": dt,
        })
        if val_macro > best_val_f1:
            best_val_f1 = val_macro
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stop at epoch {epoch} "
                      f"(best epoch {best_epoch}, val macroF1={best_val_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating on test set...")
    test_proba, test_y, test_sids = _evaluate(model, test_loader, device)
    per_ep = _per_endpoint(test_y, test_proba)
    overall = _overall(test_y, test_proba)
    print("\n=== HETERO-GNN TEST METRICS ===")
    print("  endpoint    AUROC    AUPRC    n_pos")
    for r in per_ep.itertuples(index=False):
        print(f"  {r.endpoint:<10s} {r.auroc:8.3f} {r.auprc:8.3f} {r.support:8d}")
    print(f"  overall: acc={overall['accuracy']:.3f}  "
          f"macroF1={overall['macro_f1']:.3f}  "
          f"weightedF1={overall['weighted_f1']:.3f}  "
          f"logloss={overall['log_loss']:.3f}")

    # Save
    rows = []
    val_proba_final, val_y_final, _ = _evaluate(model, val_loader, device)
    for split_name, (y_eval, proba) in [
        ("val", (val_y_final, val_proba_final)),
        ("test", (test_y, test_proba)),
    ]:
        ov = _overall(y_eval, proba)
        per = _per_endpoint(y_eval, proba)
        for r in per.itertuples(index=False):
            rows.append({"model": "hetero_gnn", "split": split_name,
                         "endpoint": r.endpoint,
                         "auroc": r.auroc, "auprc": r.auprc,
                         "support": r.support, **ov})
    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(HETERO_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    preds = pd.DataFrame({
        "subject_id": test_sids,
        "endpoint_true": pd.Series(test_y).map(
            {i: ep for ep, i in EP_TO_IDX.items()}).values,
    })
    for j, ep in enumerate(ENDPOINT_ORDER):
        preds[f"hetero_p_{ep}"] = test_proba[:, j]
    preds_path = os.path.join(HETERO_DIR, "predictions_test.csv")
    preds.to_csv(preds_path, index=False)

    pd.DataFrame(history).to_csv(
        os.path.join(HETERO_DIR, "history.csv"), index=False)
    torch.save({"state_dict": best_state, "best_epoch": best_epoch,
                "best_val_f1": best_val_f1},
                os.path.join(HETERO_DIR, "best_model.pt"))

    # Comparison figure (4-model ROC)
    base_preds = pd.read_csv(
        os.path.join(OUTPUT_DIR, "baseline", "predictions_test.csv"))
    order = base_preds["subject_id"].to_numpy()
    sid_to_idx = {int(s): i for i, s in enumerate(test_sids)}
    hetero_ordered = test_proba[[sid_to_idx[int(s)] for s in order]]
    yte_ordered = test_y[[sid_to_idx[int(s)] for s in order]]
    roc_path = os.path.join(FIGURES_DIR, "fig7_all_models_roc.png")
    _plot_roc_compare(yte_ordered, hetero_ordered, roc_path)

    print(f"\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {preds_path}")
    print(f"  {roc_path}")


if __name__ == "__main__":
    train_and_eval()
