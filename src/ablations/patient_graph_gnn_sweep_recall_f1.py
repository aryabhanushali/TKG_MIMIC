"""Same hyperparameter sweep as patient_graph_gnn_sweep.py, but selecting
on class-1 recall / F1 instead of AUROC. See xgb_survival_sweep_recall_f1.py
for why recall alone is a degenerate selection criterion at this
prevalence (0.4%-5.2%) -- confirmed the same way here before falling back
to F1. Per-epoch model selection during training still uses validation
mean AUROC@3y (unchanged, to keep the two sweeps' epoch-selection logic
identical and comparable); only the SWEEP's cross-config selection
criterion changes to F1.

Reuses the exact same coordinate-wise candidate grid as
patient_graph_gnn_sweep.py.

Output: tkg_output/sweeps/patient_graph_gnn_sweep_recall_f1.csv (screening)
        tkg_output/sweeps/patient_graph_gnn_recall_f1_best/test_metrics.csv (final, 5-seed)
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import OUTPUT_DIR
from src.tgn_model import WEIGHT_DECAY, EPOCHS, PATIENCE, _set_seed
from src.tgn_survival import (
    CAUSES, NUM_CAUSES, NUM_TIME_BINS, HORIZON_DAYS, MIN_EPOCHS,
    _make_time_bins, _deephit_nll_per_sample, _prepare_survival_targets,
    _per_cause_auroc_at_horizons,
)
from src.ablations.patient_graph_gnn import PatientConceptGNN, _prepare_patient_graph_data
from src.ablations.threshold_metrics import best_threshold_metrics

SWEEP_DIR = os.path.join(OUTPUT_DIR, "sweeps")
DEFAULTS = dict(d_model=128, n_layers=2, num_bases=4, dropout=0.15, lr=1e-3)

AXES = {
    "d_model":  [64, 128, 256, 384],
    "n_layers": [1, 2, 3, 4],
    "num_bases": [2, 4, 8, 16],
    "dropout":  [0.0, 0.15, 0.3, 0.5],
    "lr":       [3e-4, 1e-3, 3e-3, 1e-2],
}


def _labels_for(cause, h, evts, durs):
    is_pos = (evts == cause) & (durs <= h)
    survived = durs >= h
    competing = (durs < h) & (evts != cause) & (evts != "censored")
    is_neg = survived | competing
    mask = is_pos | is_neg
    y = is_pos[mask].astype(int)
    return y, mask


def _recall_f1_at_horizon(cif, sids, labels_df, time_edges, h) -> tuple:
    """Mean (max_recall, best_f1) across the 5 causes at horizon h, from a
    (n, n_causes, n_time_bins) CIF array -- mirrors _per_cause_auroc_at_horizons's
    binning but scores threshold-based metrics instead of AUROC."""
    sub = labels_df[labels_df["subject_id"].isin(sids)].set_index("subject_id").loc[sids]
    evts = sub["endpoint_type"].to_numpy()
    durs = sub["time_to_event_days"].to_numpy(dtype=float)
    bin_idx = int(np.searchsorted(time_edges, h, side="right") - 1)
    bin_idx = max(0, min(bin_idx, cif.shape[-1] - 1))
    recalls, f1s = [], []
    for i, cause in enumerate(CAUSES):
        y, mask = _labels_for(cause, h, evts, durs)
        s = cif[:, i, bin_idx][mask]
        m = best_threshold_metrics(y, s)
        if not np.isnan(m["max_recall"]):
            recalls.append(m["max_recall"])
            f1s.append(m["best_f1"])
    return (float(np.mean(recalls)) if recalls else float("nan"),
            float(np.mean(f1s)) if f1s else float("nan"))


def _train_one(d, config, seed=42, verbose=False):
    """Train PatientConceptGNN with the given hyperparameters. Per-epoch
    checkpoint selection uses validation mean AUROC@3y, unchanged from
    patient_graph_gnn_sweep.py -- returns the best checkpoint plus that
    checkpoint's val recall/F1 for the sweep's own (different) selection."""
    _set_seed(seed)
    labels_df = d["labels_df"]
    pid_to_pos = d["pid_to_pos"]
    train_sids = d["splits"]["train"]
    train_durations = labels_df.loc[
        labels_df["subject_id"].isin(train_sids), "time_to_event_days"
    ].to_numpy(dtype=np.float32)
    time_edges = _make_time_bins(train_durations, NUM_TIME_BINS)
    survival_targets = _prepare_survival_targets(labels_df, time_edges)

    n_patients = d["n_patients"]
    event_idx_all = np.zeros(n_patients, dtype=np.int64)
    duration_idx_all = np.zeros(n_patients, dtype=np.int64)
    for sid, (dur_idx, evt_idx) in survival_targets.items():
        pos = pid_to_pos[sid]
        event_idx_all[pos] = evt_idx
        duration_idx_all[pos] = dur_idx

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = PatientConceptGNN(
        n_concepts=d["n_concepts"], n_patients=d["n_patients"], n_static=d["n_static"],
        n_relations=d["n_relations"], edge_index=d["edge_index"], edge_type=d["edge_type"],
        n_classes=NUM_CAUSES * NUM_TIME_BINS,
        d_model=config["d_model"], n_layers=config["n_layers"],
        num_bases=config["num_bases"], dropout=config["dropout"],
    ).to(device)

    static_all = d["static_arr"].to(device)
    train_pos = np.array([pid_to_pos[s] for s in train_sids])
    val_pos = np.array([pid_to_pos[s] for s in d["splits"]["val"]])
    val_sids = np.array(d["splits"]["val"])

    train_event_idx = torch.tensor(event_idx_all[train_pos], dtype=torch.long, device=device)
    train_dur_idx = torch.tensor(duration_idx_all[train_pos], dtype=torch.long, device=device)
    train_pos_t = torch.tensor(train_pos, dtype=torch.long, device=device)

    counts = np.bincount(event_idx_all[train_pos], minlength=NUM_CAUSES + 1).astype(float)
    weights = np.ones_like(counts)
    weights[1:] = (counts.sum() / (NUM_CAUSES * counts[1:].clip(min=1)))
    weights = weights / weights.mean()
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

    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_metric, best_epoch, no_improve, best_state = -1.0, -1, 0, None
    for epoch in range(1, EPOCHS + 1):
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
        val_metrics = _per_cause_auroc_at_horizons(cif_val, val_sids, labels_df, time_edges, HORIZON_DAYS)
        mean3y = float(val_metrics[val_metrics["horizon_days"] == 1095]["auroc"].mean(skipna=True))
        if verbose:
            print(f"    ep {epoch:02d} val_mean_AUROC@3y={mean3y:.4f}")
        if epoch < MIN_EPOCHS:
            continue
        if mean3y > best_metric:
            best_metric, best_epoch, no_improve = mean3y, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        cif_val_best = _cif_for(model(static_all), val_pos)
    val_recall, val_f1 = _recall_f1_at_horizon(cif_val_best, val_sids, labels_df, time_edges, 1095)

    return best_metric, best_epoch, best_state, time_edges, model, device, val_recall, val_f1


def run_screening() -> pd.DataFrame:
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print("Loading graph data once (shared across all sweep configs)...")
    d = _prepare_patient_graph_data()

    rows = []
    seen = set()
    configs = [("baseline", dict(DEFAULTS))]
    for axis, values in AXES.items():
        for v in values:
            cfg = dict(DEFAULTS)
            cfg[axis] = v
            key = tuple(sorted(cfg.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append((f"{axis}={v}", cfg))

    print(f"\nScreening {len(configs)} configs (seed=42, val-only selection, recall/F1 @ 3y)...\n")
    for name, cfg in configs:
        t0 = time.time()
        best_auroc, best_epoch, _, _, _, _, val_recall, val_f1 = _train_one(d, cfg, seed=42)
        dt = time.time() - t0
        print(f"  {name:16s} d_model={cfg['d_model']:4d} n_layers={cfg['n_layers']} "
              f"num_bases={cfg['num_bases']:2d} dropout={cfg['dropout']:.2f} lr={cfg['lr']:.4f}  "
              f"-> val_mean_recall@3y={val_recall:.4f} val_mean_best_f1@3y={val_f1:.4f} "
              f"(checkpoint ep {best_epoch}, val_AUROC@3y={best_auroc:.4f}, {dt:.1f}s)")
        rows.append(dict(name=name, **cfg, val_mean_recall_3y=val_recall, val_mean_best_f1_3y=val_f1,
                          val_mean_auroc_3y=best_auroc, best_epoch=best_epoch, seconds=dt))

    result = pd.DataFrame(rows)
    out_path = os.path.join(SWEEP_DIR, "patient_graph_gnn_sweep_recall_f1.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    recall_spread = result["val_mean_recall_3y"].max() - result["val_mean_recall_3y"].min()
    print(f"\nRecall spread across all {len(result)} configs: {recall_spread:.4f} "
          f"({'confirmed degenerate -- selecting on F1 instead' if recall_spread < 0.01 else 'not degenerate, selecting on recall'})")
    return result


def best_combined_config(result: pd.DataFrame, select_col: str) -> dict:
    best = dict(DEFAULTS)
    axis_caster = {"d_model": int, "n_layers": int, "num_bases": int,
                   "dropout": float, "lr": float}
    baseline_score = result.loc[result["name"] == "baseline", select_col].iloc[0]
    print(f"\nBaseline {select_col} = {baseline_score:.4f}")
    for axis in AXES:
        pool = result[result["name"].str.startswith(f"{axis}=") | (result["name"] == "baseline")]
        top = pool.loc[pool[select_col].idxmax()]
        best[axis] = axis_caster[axis](top[axis])
        print(f"  best {axis}: {best[axis]} ({select_col}={top[select_col]:.4f}, "
              f"{'improves' if top[select_col] > baseline_score else 'no improvement'} over baseline)")
    return best


def run_final(best_cfg: dict) -> None:
    print(f"\nFinal best-combined config (selected by F1): {best_cfg}")
    out_dir = os.path.join(SWEEP_DIR, "patient_graph_gnn_recall_f1_best")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for seed in [42, 43, 44, 45, 46]:
        print(f"\n=== seed {seed} ===")
        d = _prepare_patient_graph_data()
        best_auroc, best_epoch, best_state, time_edges, model, device, _, _ = _train_one(d, best_cfg, seed=seed)
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            static_all = d["static_arr"].to(device)
            logits_flat_final = model(static_all)
            test_pos = np.array([d["pid_to_pos"][s] for s in d["splits"]["test"]])
            logits = logits_flat_final[test_pos].view(-1, NUM_CAUSES, NUM_TIME_BINS)
            probs = F.softmax(logits.reshape(logits.size(0), -1), dim=-1).view_as(logits)
            cif_test = torch.cumsum(probs, dim=-1).detach().cpu().numpy()
        test_sids = np.array(d["splits"]["test"])

        for h in HORIZON_DAYS:
            sub = d["labels_df"][d["labels_df"]["subject_id"].isin(test_sids)].set_index("subject_id").loc[test_sids]
            evts = sub["endpoint_type"].to_numpy()
            durs = sub["time_to_event_days"].to_numpy(dtype=float)
            bin_idx = int(np.searchsorted(time_edges, h, side="right") - 1)
            bin_idx = max(0, min(bin_idx, cif_test.shape[-1] - 1))
            for i, cause in enumerate(CAUSES):
                y, mask = _labels_for(cause, h, evts, durs)
                s = cif_test[:, i, bin_idx][mask]
                m = best_threshold_metrics(y, s)
                all_rows.append(dict(seed=seed, cause=cause, horizon_days=h,
                                      model="patient_graph_recall_f1_tuned", **m))
        print(f"  seed {seed}: best val_AUROC@3y={best_auroc:.4f} (epoch {best_epoch})")

    result = pd.DataFrame(all_rows)
    out_path = os.path.join(out_dir, "test_metrics.csv")
    result.to_csv(out_path, index=False)
    print(f"\n=== TUNED-FOR-F1 PATIENT-GRAPH GNN, 5-SEED TEST BEST-F1 @ 3y ===")
    print(result[result.horizon_days == 1095].groupby("cause")["best_f1"].agg(["mean", "std"]).round(4))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    screening = run_screening()
    recall_spread = screening["val_mean_recall_3y"].max() - screening["val_mean_recall_3y"].min()
    select_col = "val_mean_recall_3y" if recall_spread >= 0.01 else "val_mean_best_f1_3y"
    best_cfg = best_combined_config(screening, select_col)
    run_final(best_cfg)
