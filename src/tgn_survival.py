"""TKG temporal model with DeepHit-style competing-risks survival head.

Reuses the TKGTransformer encoder from tgn_model.py and replaces the 6-class
softmax with a discrete-time cause-specific hazard head. The loss is the
DeepHit NLL term (Lee et al. 2018); the ranking term is omitted.

Each patient is mapped to a PMF over (cause, time_bin), then cumulated to a
cumulative incidence function (CIF) per cause. Evaluation: per-cause AUROC at
1-, 3-, and 5-year horizons.
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
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import OUTPUT_DIR, FIGURES_DIR, SEED
from src.tgn_model import (
    TKGTransformer,
    PatientEventsDataset, collate,
    MAX_SEQ_LEN, D_MODEL, N_HEADS, N_LAYERS, DROPOUT,
    BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, PATIENCE,
    _set_seed, _prepare_data,
)

# Seed 42 (the default/canonical run) writes to tgn_survival/, which is what
# compare_survival.py, evaluate_stats.py, explain*.py, and make_figures.py all
# read as THE reported model. Other seeds (via TKG_SEED env var), used only
# for multi-seed mean+/-std robustness reporting, write to their own
# tgn_survival_seed{N}/ so they never overwrite the canonical run.
MODEL_DIR = os.path.join(OUTPUT_DIR, "tgn_survival" if SEED == 42 else f"tgn_survival_seed{SEED}")
MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")

# All 5 seeds (42-46) early-stopped with "best epoch" = 1 or 2 by val mean
# AUROC@3y. A GNNExplainer fidelity check on that checkpoint (explain_gnn.py)
# found its "important" events statistically indistinguishable from randomly
# chosen ones (sufficiency/comprehensiveness KL ~= random-baseline KL) -- i.e.
# the checkpoint being selected is too undertrained for its internal
# event-importance signal to be meaningful, even though its aggregate
# discrimination is already competitive. MIN_EPOCHS makes epochs before this
# floor ineligible for "best" checkpoint selection, so early-stopping cannot
# lock in a checkpoint before the model has had a real minimum of training.
MIN_EPOCHS = 15

# Competing-risks setup: 5 cause-specific endpoints + censored
CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]
CAUSE_TO_IDX = {c: i + 1 for i, c in enumerate(CAUSES)}
NUM_CAUSES = len(CAUSES)
NUM_TIME_BINS = 12

HORIZON_DAYS = [365, 1095, 1825]


def _make_time_bins(train_durations: np.ndarray, num_bins: int) -> np.ndarray:
    """Quantile-based time bin edges from training durations.
    Returns array of length num_bins+1: edges[0] = 0, edges[-1] = max+epsilon."""
    qs = np.linspace(0, 1, num_bins + 1)
    edges = np.quantile(train_durations[train_durations > 0], qs)
    edges[0] = 0.0
    edges[-1] = float(train_durations.max()) + 1.0
    # ensure monotone
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1.0
    return edges


def _discretize(duration: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map a continuous duration to a bin index 0..num_bins-1."""
    idx = np.searchsorted(edges, duration, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2).astype(np.int64)


class TKGSurvivalNet(nn.Module):
    """TKGTransformer encoder + cause-specific discrete-time hazard head.

    The encoder's classification head is repurposed to emit
    ``num_causes * num_time_bins`` logits, which are then reshaped into the
    (cause, time) grid used by the DeepHit NLL.
    """
    def __init__(self, n_concepts, n_edge_types, n_static,
                 num_causes=NUM_CAUSES, num_time_bins=NUM_TIME_BINS,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.encoder = TKGTransformer(
            n_concepts=n_concepts, n_edge_types=n_edge_types,
            n_static=n_static, n_classes=num_causes * num_time_bins,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout,
        )
        self.num_causes = num_causes
        self.num_time_bins = num_time_bins

    def forward(self, concept_idx, edge_type_idx, t, v_norm, v_present,
                static, mask, return_attention: bool = False):
        if return_attention:
            flat, attn = self.encoder(
                concept_idx, edge_type_idx, t,
                v_norm, v_present, static, mask, return_attention=True,
            )
            return flat.view(-1, self.num_causes, self.num_time_bins), attn
        flat = self.encoder(concept_idx, edge_type_idx, t,
                             v_norm, v_present, static, mask)
        return flat.view(-1, self.num_causes, self.num_time_bins)


def deephit_nll(logits: torch.Tensor,
                duration_idx: torch.Tensor,
                event_idx: torch.Tensor) -> torch.Tensor:
    """
    logits:       (B, C, T) -- raw logits over (cause, time_bin)
    duration_idx: (B,)      -- time bin of event/censoring (0..T-1)
    event_idx:    (B,)      -- 0=censored, 1..C=cause index
    """
    B, C, T = logits.shape
    flat = logits.reshape(B, C * T)
    log_probs = F.log_softmax(flat, dim=-1).reshape(B, C, T)
    probs = torch.exp(log_probs)                                # (B, C, T)
    cum_per_cause = torch.cumsum(probs, dim=-1)                 # (B, C, T)
    cum_total = cum_per_cause.sum(dim=1).clamp(max=1.0 - 1e-7)  # (B, T)
    surv = (1.0 - cum_total).clamp(min=1e-7)                    # P(no event by t)
    log_surv = torch.log(surv)                                  # (B, T)

    is_censored = (event_idx == 0)
    cause_for_event = (event_idx - 1).clamp(min=0)
    obs_log_prob = log_probs.gather(
        1, cause_for_event.view(B, 1, 1).expand(-1, 1, T)
    ).squeeze(1)                                                # (B, T)
    obs_log_prob_at = obs_log_prob.gather(1, duration_idx.view(B, 1)).squeeze(1)
    log_surv_at = log_surv.gather(1, duration_idx.view(B, 1)).squeeze(1)

    loss = torch.where(is_censored, -log_surv_at, -obs_log_prob_at)
    return loss.mean()


def _prepare_survival_targets(labels_df: pd.DataFrame, time_edges: np.ndarray):
    """Returns dict: subject_id -> (duration_idx, event_idx)."""
    out = {}
    for r in labels_df.itertuples(index=False):
        sid = int(r.subject_id)
        dur = float(r.time_to_event_days) if pd.notna(r.time_to_event_days) else 0.0
        dur = max(dur, 0.0)
        # event_idx: 0 = censored, else CAUSE_TO_IDX[label]. An unrecognized
        # endpoint_type must fail loudly, not silently get counted as
        # censored -- that would quietly corrupt training labels instead of
        # surfacing a real data problem.
        if r.endpoint_type == "censored":
            event_idx = 0
        elif r.endpoint_type in CAUSE_TO_IDX:
            event_idx = CAUSE_TO_IDX[r.endpoint_type]
        else:
            raise ValueError(
                f"subject_id {sid}: unrecognized endpoint_type "
                f"{r.endpoint_type!r} (expected 'censored' or one of {CAUSES})"
            )
        d_idx = int(_discretize(np.array([dur]), time_edges)[0])
        out[sid] = (d_idx, event_idx)
    return out


def _evaluate_survival(model, loader, device):
    """Run the model on `loader` and return per-patient cumulative incidence
    functions (CIF) of shape (N, C, T)."""
    model.eval()
    cif_all, sid_all = [], []
    with torch.no_grad():
        for batch in loader:
            for k in ("concept_idx", "edge_type_idx", "t",
                     "v_norm", "v_present", "static", "mask"):
                batch[k] = batch[k].to(device)
            logits = model(batch["concept_idx"], batch["edge_type_idx"],
                            batch["t"], batch["v_norm"], batch["v_present"],
                            batch["static"], batch["mask"])
            B, C, T = logits.shape
            flat = logits.reshape(B, C * T)
            probs = F.softmax(flat, dim=-1).reshape(B, C, T)
            cif = torch.cumsum(probs, dim=-1).cpu().numpy()      # (B, C, T)
            cif_all.append(cif)
            sid_all.append(batch["sid"].numpy())
    cif_arr = np.concatenate(cif_all, axis=0)
    sids = np.concatenate(sid_all, axis=0)
    return cif_arr, sids


def _per_cause_auroc_at_horizons(cif: np.ndarray, sids: np.ndarray,
                                  labels_df: pd.DataFrame,
                                  time_edges: np.ndarray,
                                  horizons: list[int]) -> pd.DataFrame:
    """Per-cause AUROC at each horizon h.

    For cause c at horizon h: positive = (observed cause == c AND duration <= h);
    negative = survived past h, OR had a *competing* observed event before h
    (a competing event means cause c did not occur by h). Only patients
    administratively censored before h are dropped (status unknown; IPCW-free).
    """
    durs = dict(zip(labels_df["subject_id"], labels_df["time_to_event_days"]))
    evts = dict(zip(labels_df["subject_id"], labels_df["endpoint_type"]))
    rows = []
    for h_days in horizons:
        h_bin = int(_discretize(np.array([h_days]), time_edges)[0])
        for c_idx, cause in enumerate(CAUSES):
            y, p = [], []
            for i, sid in enumerate(sids):
                d = durs.get(int(sid), np.nan)
                e = evts.get(int(sid), "censored")
                if pd.isna(d):
                    continue
                if e == cause and d <= h_days:
                    y.append(1); p.append(cif[i, c_idx, h_bin])
                elif d >= h_days:
                    y.append(0); p.append(cif[i, c_idx, h_bin])
                elif e != cause and e != "censored":   # competing event before h
                    y.append(0); p.append(cif[i, c_idx, h_bin])
            y, p = np.array(y), np.array(p)
            if y.sum() == 0 or y.sum() == len(y):
                rows.append({"cause": cause, "horizon_days": h_days,
                             "auroc": np.nan, "auprc": np.nan,
                             "n_pos": int(y.sum()), "n": len(y)})
                continue
            rows.append({
                "cause": cause, "horizon_days": h_days,
                "auroc": roc_auc_score(y, p),
                "auprc": average_precision_score(y, p),
                "n_pos": int(y.sum()), "n": len(y),
            })
    return pd.DataFrame(rows)


def train_and_eval() -> None:
    _set_seed()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data via tgn_model._prepare_data ...")
    # The 6-class label_by_sid returned by _prepare_data is discarded here;
    # survival targets are built below from labels_df.
    (events_by_sid, static_by_sid, _,
     splits, n_concepts, n_edge_types, n_static, labels_df) = _prepare_data()

    train_sids = splits["train"]
    train_durations = labels_df.loc[
        labels_df["subject_id"].isin(train_sids), "time_to_event_days"
    ].to_numpy(dtype=np.float32)
    time_edges = _make_time_bins(train_durations, NUM_TIME_BINS)
    print(f"  time bin edges (days): {time_edges.round(1).tolist()}")

    # The PatientEventsDataset 'label' slot is repurposed to event_idx;
    # duration_idx is read from a parallel lookup in the training loop.
    survival_targets = _prepare_survival_targets(labels_df, time_edges)
    label_by_sid = {sid: int(t[1]) for sid, t in survival_targets.items()}
    duration_by_sid = {sid: int(t[0]) for sid, t in survival_targets.items()}

    def _make_loader(sids, shuffle):
        ds = PatientEventsDataset(sids, events_by_sid, static_by_sid,
                                    label_by_sid, max_len=MAX_SEQ_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                            collate_fn=collate, num_workers=0)

    train_loader = _make_loader(splits["train"], shuffle=True)
    val_loader   = _make_loader(splits["val"],   shuffle=False)
    test_loader  = _make_loader(splits["test"],  shuffle=False)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  device: {device}")

    model = TKGSurvivalNet(
        n_concepts=n_concepts, n_edge_types=n_edge_types,
        n_static=n_static, num_causes=NUM_CAUSES,
        num_time_bins=NUM_TIME_BINS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    # Inverse-frequency weighting (training cohort only); censored stays at 1
    counts = np.zeros(NUM_CAUSES + 1)
    for sid in splits["train"]:
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

    optim = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_metric = -1.0
    best_epoch = -1
    no_improve = 0
    history = []
    best_state = None

    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE}, "
          f"min_epochs={MIN_EPOCHS} before a checkpoint is eligible as 'best')...")
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
            evt_idx = torch.tensor(
                [label_by_sid[s] for s in sids], dtype=torch.long, device=device)
            dur_idx = torch.tensor(
                [duration_by_sid[s] for s in sids], dtype=torch.long, device=device)
            optim.zero_grad()
            logits = model(batch["concept_idx"], batch["edge_type_idx"],
                            batch["t"], batch["v_norm"], batch["v_present"],
                            batch["static"], batch["mask"])
            loss = weighted_deephit_nll(logits, dur_idx, evt_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        train_loss /= max(n_batches, 1)

        cif_val, sids_val = _evaluate_survival(model, val_loader, device)
        val_metrics = _per_cause_auroc_at_horizons(
            cif_val, sids_val, labels_df, time_edges, HORIZON_DAYS)
        mean3y = float(val_metrics[
            val_metrics["horizon_days"] == 1095
        ]["auroc"].mean(skipna=True))
        dt = time.time() - t0
        print(f"  ep {epoch:02d}  loss={train_loss:.4f}  "
              f"val_mean_AUROC@3y={mean3y:.4f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_mean_auroc_3y": mean3y, "time_s": dt})
        if epoch < MIN_EPOCHS:
            continue   # too early to be eligible as "best"; keep training
        if mean3y > best_metric:
            best_metric = mean3y; best_epoch = epoch; no_improve = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stop at epoch {epoch} "
                      f"(best epoch {best_epoch}, val mean AUROC@3y={best_metric:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating on test set...")
    cif_test, sids_test = _evaluate_survival(model, test_loader, device)
    test_metrics = _per_cause_auroc_at_horizons(
        cif_test, sids_test, labels_df, time_edges, HORIZON_DAYS)
    print("\n=== TGN-SURVIVAL TEST METRICS ===")
    print("  Per-cause AUROC / AUPRC at 1y / 3y / 5y horizons:")
    pivot_auc = test_metrics.pivot(
        index="cause", columns="horizon_days", values="auroc").round(3)
    pivot_pr = test_metrics.pivot(
        index="cause", columns="horizon_days", values="auprc").round(3)
    print("\nAUROC:\n" + pivot_auc.to_string())
    print("\nAUPRC:\n" + pivot_pr.to_string())

    test_metrics.to_csv(os.path.join(MODEL_DIR, "test_metrics.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(MODEL_DIR, "history.csv"), index=False)

    horizon_bins = [int(_discretize(np.array([h]), time_edges)[0])
                     for h in HORIZON_DAYS]
    preds_rows = []
    durs = dict(zip(labels_df["subject_id"], labels_df["time_to_event_days"]))
    evts = dict(zip(labels_df["subject_id"], labels_df["endpoint_type"]))
    for i, sid in enumerate(sids_test):
        row = {"subject_id": int(sid),
               "endpoint_true": evts.get(int(sid), "censored"),
               "duration_days": float(durs.get(int(sid), np.nan))}
        for h_days, h_bin in zip(HORIZON_DAYS, horizon_bins):
            for c_idx, cause in enumerate(CAUSES):
                row[f"cif_{cause}_at_{h_days}d"] = float(cif_test[i, c_idx, h_bin])
        preds_rows.append(row)
    pd.DataFrame(preds_rows).to_csv(
        os.path.join(MODEL_DIR, "predictions_test.csv"), index=False)
    torch.save({"state_dict": best_state, "time_edges": time_edges,
                 "best_epoch": best_epoch, "best_val_metric": best_metric},
                 os.path.join(MODEL_DIR, "best_model.pt"))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"MI": "#d62728", "Stroke": "#9467bd", "HF": "#ff7f0e",
              "AF": "#1f77b4", "PAD": "#2ca02c"}
    for cause in CAUSES:
        sub = test_metrics[test_metrics["cause"] == cause]
        ax.plot(sub["horizon_days"], sub["auroc"], "o-",
                label=cause, color=colors[cause], linewidth=2)
    ax.set_xlabel("Horizon (days from index)")
    ax.set_ylabel("Time-dependent AUROC (test)")
    ax.set_title("TGN-Survival: per-cause AUROC by horizon",
                 fontweight="bold")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout()
    print(f"\nSaved:")
    print(f"  {os.path.join(MODEL_DIR, 'test_metrics.csv')}")
    print(f"  {os.path.join(MODEL_DIR, 'predictions_test.csv')}")
    print(f"  {os.path.join(MODEL_DIR, 'history.csv')}")
    print(f"  {os.path.join(MODEL_DIR, 'best_model.pt')}")
    # This paper figure reports THE canonical (seed=42) model; other seeds are
    # numeric-only robustness runs and must not overwrite it.
    if SEED == 42:
        fig.savefig(os.path.join(FIGURES_DIR, "fig8_tgn_survival_per_cause.png"),
                    dpi=300, bbox_inches="tight")
        print(f"  {os.path.join(FIGURES_DIR, 'fig8_tgn_survival_per_cause.png')}")
    plt.close(fig)


def _deephit_nll_per_sample(logits, duration_idx, event_idx):
    """Per-sample DeepHit NLL (no reduction)."""
    B, C, T = logits.shape
    flat = logits.reshape(B, C * T)
    log_probs = F.log_softmax(flat, dim=-1).reshape(B, C, T)
    probs = torch.exp(log_probs)
    cum_per_cause = torch.cumsum(probs, dim=-1)
    cum_total = cum_per_cause.sum(dim=1).clamp(max=1.0 - 1e-7)
    surv = (1.0 - cum_total).clamp(min=1e-7)
    log_surv = torch.log(surv)

    is_censored = (event_idx == 0)
    cause_for_event = (event_idx - 1).clamp(min=0)
    obs_log_prob = log_probs.gather(
        1, cause_for_event.view(B, 1, 1).expand(-1, 1, T)
    ).squeeze(1)
    obs_log_prob_at = obs_log_prob.gather(1, duration_idx.view(B, 1)).squeeze(1)
    log_surv_at = log_surv.gather(1, duration_idx.view(B, 1)).squeeze(1)
    return torch.where(is_censored, -log_surv_at, -obs_log_prob_at)


if __name__ == "__main__":
    train_and_eval()
