"""A genuinely patient-in-the-graph GNN: patients are real nodes, connected
to other patients only through shared concepts, updated by real multi-hop
message passing. This is the thing hetero_gnn.py never actually tested --
that file only ever enriched CONCEPT embeddings with graph structure; every
patient was still a flat, structurally-isolated sequence with zero
connection to any other patient. A sequence model cannot do what this can:
let one patient's prediction be informed by patterns learned from OTHER
patients who share their concepts, multiple hops away.

Graph:
  - Concept nodes (train-restricted vocabulary + UNK, same convention as
    tgn_model.py) and patient nodes, sharing one combined index space
    (concepts first, patients after).
  - patient -> concept / concept -> patient edges, deduplicated per
    (patient, concept) pair, split into 3 recency relations by the most
    recent occurrence (<=90d / 90-730d / >730d before index) so SOME timing
    signal survives the collapse from a sequence into a graph.
  - concept -isA-> concept (ICD hierarchy / drug class / lab category,
    from build_ontology.py) and concept -cooccurs-> concept (from
    build_cooccurrence.py, train-patients-only, so no val/test structure
    leaks into the graph itself).

Training is full-batch, not mini-batched: one forward pass over the WHOLE
graph per epoch (every patient node is structurally present, standard for
transductive GNNs), loss computed only on the training patients' positions.
This is both the computationally correct way to train a GNN like this (the
previous per-mini-batch design recomputed the full graph ~184 times per
epoch for no benefit) and clearer about the actual assumption being made:
test/val patients' OWN pre-index facts (already time-windowed and safe)
shape their node's position in the graph, but their LABELS are never used,
and no test/val patient gets a free learnable parameter of their own -- a
patient's initial node state is a deterministic function of their own
static features, not a per-patient embedding lookup, so nothing about a
specific test patient is *fit* during training beyond what message-passing
structurally requires.

Same DeepHit competing-risks head, time bins, corrected horizon evaluation,
and MIN_EPOCHS floor as tgn_survival.py, imported directly for a fair,
literally-identical evaluation protocol.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from src.config import OUTPUT_DIR, FIGURES_DIR, SEED, read_events_table
from src.tgn_model import D_MODEL, N_LAYERS, DROPOUT, LR, WEIGHT_DECAY, EPOCHS, PATIENCE, _set_seed
from src.tgn_survival import (
    CAUSES, NUM_CAUSES, NUM_TIME_BINS, HORIZON_DAYS, MIN_EPOCHS,
    _make_time_bins, _discretize, _deephit_nll_per_sample,
    _prepare_survival_targets, _per_cause_auroc_at_horizons,
)

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
_dirname = "patient_gnn_survival" if os.environ.get("PATIENT_GNN_TIMING", "1").lower() not in ("0", "false", "no") \
    else "patient_gnn_survival_notiming"
MODEL_DIR = os.path.join(OUTPUT_DIR, _dirname if SEED == 42 else f"{_dirname}_seed{SEED}")

RECENCY_BINS = [90, 730]   # days before index: <=90 "recent", 90-730 "mid", >730 "old"
N_RGCN_LAYERS = 2   # 2 hops: patient -> concept -> patient reaches a same-concept neighbor

# 2x2 ablation (patient-nodes x timing information): with PATIENT_GNN_TIMING=0,
# every patient-concept edge collapses to a single relation regardless of
# when the event occurred, isolating whether the model's advantage comes
# from patient-to-patient connectivity alone or needs the residual recency
# signal on top of it. Everything else (architecture, co-occurrence edges,
# ontology edges, training procedure) is identical to the timing-on run.
USE_TIMING = os.environ.get("PATIENT_GNN_TIMING", "1").lower() not in ("0", "false", "no")


def _recency_bucket(days_before_index: np.ndarray) -> np.ndarray:
    """days_before_index >= 0 (0 = right at index date). Returns 0/1/2,
    or always 0 if USE_TIMING is disabled (single relation, no timing)."""
    if not USE_TIMING:
        return np.zeros_like(days_before_index, dtype=np.int64)
    b = np.zeros_like(days_before_index, dtype=np.int64)
    b[(days_before_index > RECENCY_BINS[0]) & (days_before_index <= RECENCY_BINS[1])] = 1
    b[days_before_index > RECENCY_BINS[1]] = 2
    return b


class PatientConceptGNN(nn.Module):
    def __init__(self, n_concepts, n_patients, n_static, n_relations,
                 edge_index, edge_type, n_classes,
                 d_model=D_MODEL, n_layers=N_RGCN_LAYERS, num_bases=4, dropout=DROPOUT):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_patients = n_patients
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

        self.concept_emb = nn.Embedding(n_concepts, d_model)
        nn.init.normal_(self.concept_emb.weight, std=0.02)
        # Patient nodes get NO free learnable per-patient parameter -- their
        # initial state is a deterministic function of their own static
        # features, so nothing patient-specific is being "memorized" other
        # than what message-passing structurally requires.
        self.patient_init = nn.Sequential(
            nn.Linear(n_static, d_model), nn.GELU(), nn.LayerNorm(d_model))

        self.rgcn = nn.ModuleList([
            RGCNConv(d_model, d_model, num_relations=n_relations, num_bases=num_bases)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes))

    def forward(self, static_all: torch.Tensor) -> torch.Tensor:
        """static_all: (n_patients, n_static) for every patient, in the same
        order as the patient block of the combined node index."""
        x0 = torch.cat([self.concept_emb.weight, self.patient_init(static_all)], dim=0)
        x = x0
        for layer in self.rgcn:
            x = layer(x, self.edge_index, self.edge_type)
            x = F.gelu(x)
            x = self.dropout(x)
        x = self.norm(x + x0)
        patient_repr = x[self.n_concepts:]
        return self.head(patient_repr)   # (n_patients, n_classes)


# --------------------------------------------------------------------------- #
# Data / graph construction                                                   #
# --------------------------------------------------------------------------- #
def _prepare_patient_graph_data():
    print("Loading modeling artifacts + ontology + co-occurrence...")
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    static = pd.read_csv(os.path.join(MODELING_DIR, "static_features.csv"))
    events = read_events_table()
    nodes_v2 = pd.read_csv(os.path.join(OUTPUT_DIR, "node_index_v2.csv"))
    ontology = pd.read_csv(os.path.join(OUTPUT_DIR, "ontology_edges.csv"))
    cooccur = pd.read_csv(os.path.join(OUTPUT_DIR, "cooccurrence_edges.csv"))

    concept_nodes = nodes_v2[nodes_v2["fact_type"] != "patient"].copy().reset_index(drop=True)
    # idx 0 = UNK; concepts 1..n_concepts-1
    concept_nodes["gnn_idx"] = np.arange(1, len(concept_nodes) + 1, dtype=np.int64)
    nidx_to_gnn = dict(zip(concept_nodes["node_idx"], concept_nodes["gnn_idx"]))
    n_concepts = len(concept_nodes) + 1

    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    train_concepts_global = set(
        events.loc[events["subject_id"].isin(train_ids), "concept_node_idx"].unique())

    # Patient order fixes each patient's position in the combined index space.
    patient_order = labels["subject_id"].tolist()
    pid_to_pos = {int(sid): i for i, sid in enumerate(patient_order)}
    n_patients = len(patient_order)

    events = events.copy()
    events["c_emb_idx"] = events["concept_node_idx"].map(nidx_to_gnn).astype("Int64")
    events = events.dropna(subset=["c_emb_idx"])
    events["c_emb_idx"] = events["c_emb_idx"].astype(np.int64)
    is_train_concept = events["concept_node_idx"].isin(train_concepts_global)
    n_oov = int((~is_train_concept).sum())
    events.loc[~is_train_concept, "c_emb_idx"] = 0
    print(f"  train-only concept restriction: {n_oov:,} events mapped to UNK "
          f"({n_oov / max(len(events), 1) * 100:.2f}%)")

    # --- patient <-> concept edges, deduped + recency-bucketed ---
    events["days_before_index"] = -events["relative_days"]   # relative_days <= 0
    most_recent = (events.groupby(["subject_id", "c_emb_idx"])["days_before_index"]
                    .min().reset_index())   # min days-before-index = most recent
    most_recent = most_recent[most_recent["subject_id"].isin(pid_to_pos)]
    most_recent["pat_pos"] = most_recent["subject_id"].map(pid_to_pos).astype(np.int64)
    most_recent["recency"] = _recency_bucket(most_recent["days_before_index"].to_numpy())
    print(f"  patient-concept pairs (deduped, all splits): {len(most_recent):,}")
    for b, name in enumerate(["recent (<=90d)", "mid (90-730d)", "old (>730d)"]):
        print(f"    {name}: {(most_recent['recency'] == b).sum():,}")

    pc_concept_idx = most_recent["c_emb_idx"].to_numpy()
    pc_patient_node = most_recent["pat_pos"].to_numpy() + n_concepts   # patients come after concepts
    pc_recency = most_recent["recency"].to_numpy()

    # Relation layout:
    #   0,1,2       = patient -> concept, by recency bucket
    #   3,4,5       = concept -> patient (reverse), by recency bucket
    #   6..6+2K-1   = ontology isA edges: build_ontology.py writes only the
    #                 child->parent direction, so the reverse parent->child
    #                 edges added below (as a separate relation id range) are
    #                 what actually makes this graph bidirectional
    #   last 2      = co-occurrence, symmetric (same relation id both ways;
    #                 note build_cooccurrence.py's top-K-per-concept cap is
    #                 applied per source concept independently, so after both
    #                 directions are added here a concept's true co-occurrence
    #                 degree can exceed K -- a deliberate "union of top-K
    #                 neighbors" graph, not a mutual-K-NN graph)
    onto_edge_types = sorted(ontology["edge_type"].unique().tolist())
    etype_to_idx = {e: i for i, e in enumerate(onto_edge_types)}
    n_onto = len(onto_edge_types)

    REL_PC_BASE = 0            # 3 relations: 0,1,2
    REL_CP_BASE = 3            # 3 relations: 3,4,5
    REL_ONTO_BASE = 6          # 2*n_onto relations
    REL_COOCCUR = REL_ONTO_BASE + 2 * n_onto   # 1 relation, used both directions
    n_relations = REL_COOCCUR + 1

    src_list, dst_list, rel_list = [], [], []

    # patient -> concept
    src_list.append(pc_patient_node); dst_list.append(pc_concept_idx)
    rel_list.append(REL_PC_BASE + pc_recency)
    # concept -> patient (reverse)
    src_list.append(pc_concept_idx); dst_list.append(pc_patient_node)
    rel_list.append(REL_CP_BASE + pc_recency)

    # ontology isA (+ reverse), mapped into the concept index space
    ontology = ontology.copy()
    ontology["src_gnn"] = ontology["src_node_idx"].map(nidx_to_gnn)
    ontology["dst_gnn"] = ontology["dst_node_idx"].map(nidx_to_gnn)
    ontology = ontology.dropna(subset=["src_gnn", "dst_gnn"])
    ontology["etype_idx"] = ontology["edge_type"].map(etype_to_idx).astype(np.int64)
    o_src = ontology["src_gnn"].to_numpy(dtype=np.int64)
    o_dst = ontology["dst_gnn"].to_numpy(dtype=np.int64)
    o_rel = ontology["etype_idx"].to_numpy(dtype=np.int64)
    src_list.append(o_src); dst_list.append(o_dst); rel_list.append(REL_ONTO_BASE + o_rel)
    src_list.append(o_dst); dst_list.append(o_src); rel_list.append(REL_ONTO_BASE + n_onto + o_rel)

    # co-occurrence (train-only-derived), symmetric same relation id
    cooccur = cooccur.copy()
    cooccur["src_gnn"] = cooccur["src_concept_node_idx"].map(nidx_to_gnn)
    cooccur["dst_gnn"] = cooccur["dst_concept_node_idx"].map(nidx_to_gnn)
    cooccur = cooccur.dropna(subset=["src_gnn", "dst_gnn"])
    c_src = cooccur["src_gnn"].to_numpy(dtype=np.int64)
    c_dst = cooccur["dst_gnn"].to_numpy(dtype=np.int64)
    src_list.append(c_src); dst_list.append(c_dst)
    rel_list.append(np.full(len(c_src), REL_COOCCUR, dtype=np.int64))
    src_list.append(c_dst); dst_list.append(c_src)
    rel_list.append(np.full(len(c_dst), REL_COOCCUR, dtype=np.int64))

    edge_index = np.stack([np.concatenate(src_list), np.concatenate(dst_list)])
    edge_type = np.concatenate(rel_list)
    print(f"  combined graph: {n_concepts + n_patients:,} nodes "
          f"({n_concepts:,} concepts + {n_patients:,} patients), "
          f"{edge_index.shape[1]:,} directed edges, {n_relations} relations")

    static_cols = ["age_at_index", "cci_score", "num_cardiometa_conditions",
                   "had_icu_stay", "female"]
    static_train = static[static["subject_id"].isin(train_ids)][static_cols]
    mu, sd = static_train.mean(), static_train.std().replace(0, 1.0)
    static_norm = (static.set_index("subject_id").loc[patient_order, static_cols] - mu) / sd
    static_arr = static_norm.to_numpy(dtype=np.float32)

    return {
        "edge_index": torch.from_numpy(edge_index).long(),
        "edge_type": torch.from_numpy(edge_type).long(),
        "n_concepts": n_concepts,
        "n_patients": n_patients,
        "n_relations": n_relations,
        "n_static": len(static_cols),
        "static_arr": torch.from_numpy(static_arr).float(),
        "patient_order": patient_order,
        "pid_to_pos": pid_to_pos,
        "labels_df": labels,
        "splits": {s: labels.loc[labels["split"] == s, "subject_id"].tolist()
                   for s in ("train", "val", "test")},
    }


# --------------------------------------------------------------------------- #
# Train + evaluate (full-batch: one forward pass over the whole graph/epoch)  #
# --------------------------------------------------------------------------- #
def train_and_eval() -> None:
    _set_seed()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    d = _prepare_patient_graph_data()
    labels_df = d["labels_df"]
    pid_to_pos = d["pid_to_pos"]

    train_sids = d["splits"]["train"]
    train_durations = labels_df.loc[
        labels_df["subject_id"].isin(train_sids), "time_to_event_days"
    ].to_numpy(dtype=np.float32)
    time_edges = _make_time_bins(train_durations, NUM_TIME_BINS)
    print(f"  time bin edges (days): {time_edges.round(1).tolist()}")

    survival_targets = _prepare_survival_targets(labels_df, time_edges)
    # Index by POSITION in patient_order, not subject_id, since the model
    # outputs one row per position.
    n_patients = d["n_patients"]
    event_idx_all = np.zeros(n_patients, dtype=np.int64)
    duration_idx_all = np.zeros(n_patients, dtype=np.int64)
    for sid, (dur_idx, evt_idx) in survival_targets.items():
        pos = pid_to_pos[sid]
        event_idx_all[pos] = evt_idx
        duration_idx_all[pos] = dur_idx

    # MPS hung (near-zero CPU usage, no progress) on this graph's scale --
    # 2.74M edges, 19 relations, num_bases=4 basis decomposition -- even
    # though a small isolated RGCNConv test on MPS worked fine. CPU is the
    # safe choice here; full-batch training (one pass/epoch, not per-batch)
    # keeps this tractable despite no GPU acceleration.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"  device: {device}")

    model = PatientConceptGNN(
        n_concepts=d["n_concepts"], n_patients=d["n_patients"], n_static=d["n_static"],
        n_relations=d["n_relations"], edge_index=d["edge_index"], edge_type=d["edge_type"],
        n_classes=NUM_CAUSES * NUM_TIME_BINS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    static_all = d["static_arr"].to(device)
    train_pos = np.array([pid_to_pos[s] for s in train_sids])
    val_pos = np.array([pid_to_pos[s] for s in d["splits"]["val"]])
    test_pos = np.array([pid_to_pos[s] for s in d["splits"]["test"]])

    train_event_idx = torch.tensor(event_idx_all[train_pos], dtype=torch.long, device=device)
    train_dur_idx = torch.tensor(duration_idx_all[train_pos], dtype=torch.long, device=device)
    train_pos_t = torch.tensor(train_pos, dtype=torch.long, device=device)

    counts = np.bincount(event_idx_all[train_pos], minlength=NUM_CAUSES + 1).astype(float)
    weights = np.ones_like(counts)
    weights[1:] = (counts.sum() / (NUM_CAUSES * counts[1:].clip(min=1)))
    weights = weights / weights.mean()
    print(f"  event weights (0..K): {weights.round(3).tolist()}")
    sample_weight_by_event = torch.tensor(weights, dtype=torch.float32, device=device)

    def weighted_deephit_nll(logits_flat, dur_idx, evt_idx):
        logits = logits_flat.view(-1, NUM_CAUSES, NUM_TIME_BINS)
        per_sample = _deephit_nll_per_sample(logits, dur_idx, evt_idx)
        w = sample_weight_by_event[evt_idx]
        return (per_sample * w).mean()

    def _cif_for(logits_flat, positions):
        logits = logits_flat[positions].view(-1, NUM_CAUSES, NUM_TIME_BINS)
        probs = F.softmax(logits.reshape(logits.size(0), -1), dim=-1).view_as(logits)
        return torch.cumsum(probs, dim=-1).detach().cpu().numpy()

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_metric, best_epoch, no_improve = -1.0, -1, 0
    history, best_state = [], None

    print(f"\nTraining for up to {EPOCHS} epochs (full-batch, patience={PATIENCE}, "
          f"min_epochs={MIN_EPOCHS})...")
    print("Selection metric: mean per-cause AUROC at 3-yr horizon on val set\n")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        optim.zero_grad()
        logits_flat = model(static_all)
        loss = weighted_deephit_nll(logits_flat[train_pos_t], train_dur_idx, train_event_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_flat_eval = model(static_all)
            cif_val = _cif_for(logits_flat_eval, val_pos)
        val_sids = np.array(d["splits"]["val"])
        val_metrics = _per_cause_auroc_at_horizons(cif_val, val_sids, labels_df, time_edges, HORIZON_DAYS)
        mean3y = float(val_metrics[val_metrics["horizon_days"] == 1095]["auroc"].mean(skipna=True))
        dt = time.time() - t0
        print(f"  ep {epoch:02d}  loss={float(loss.item()):.4f}  "
              f"val_mean_AUROC@3y={mean3y:.4f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": float(loss.item()),
                         "val_mean_auroc_3y": mean3y, "time_s": dt})
        if epoch < MIN_EPOCHS:
            continue
        if mean3y > best_metric:
            best_metric, best_epoch, no_improve = mean3y, epoch, 0
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
    model.eval()
    with torch.no_grad():
        logits_flat_final = model(static_all)
        cif_test = _cif_for(logits_flat_final, test_pos)
    test_sids = np.array(d["splits"]["test"])
    test_metrics = _per_cause_auroc_at_horizons(cif_test, test_sids, labels_df, time_edges, HORIZON_DAYS)
    print("\n=== PATIENT-GRAPH GNN TEST METRICS ===")
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
