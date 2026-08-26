"""Hyperparameter sweep for the patient-graph GNN (src/ablations/
patient_graph_gnn.py), to answer a specific question raised in review: is
this model's mediocre showing (Section 6.3, 8.6) because the architecture
doesn't fit the problem, or because nothing about it was ever tuned?

Coordinate-wise sweep from the shipped defaults (d_model=128, n_layers=2,
num_bases=4, dropout=0.15, lr=1e-3) across one axis at a time -- cheaper and
more diagnostic than a full grid, since the question is "does turning this
knob help at all," not "find the global optimum." Screening uses seed 42
only (this architecture trains in ~1-2 minutes, so this is still fast); the
single best value per axis is then combined into one config and re-run
across all 5 standard seeds (42-46) for a result directly comparable to the
rest of this study's multi-seed tables.

Selection metric: validation mean AUROC at the 3-year horizon -- the exact
same metric already used for this model's own epoch-level checkpoint
selection (src/ablations/patient_graph_gnn.py), so no new selection
criterion is introduced and no test-set information is ever touched during
the sweep. Only the final, single best-combined config's TEST result is
reported, once, at the end -- same discipline as every other model in this
study (Section 11).

Output: tkg_output/sweeps/patient_graph_gnn_sweep.csv (screening results)
        tkg_output/sweeps/patient_graph_gnn_best/test_metrics.csv (final,
        best-combined config, 5-seed test result)
"""
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import OUTPUT_DIR, SEED
from src.tgn_model import LR as DEFAULT_LR, WEIGHT_DECAY, EPOCHS, PATIENCE, _set_seed
from src.tgn_survival import (
    CAUSES, NUM_CAUSES, NUM_TIME_BINS, HORIZON_DAYS, MIN_EPOCHS,
    _make_time_bins, _deephit_nll_per_sample, _prepare_survival_targets,
    _per_cause_auroc_at_horizons,
)
from src.ablations.patient_graph_gnn import PatientConceptGNN, _prepare_patient_graph_data

SWEEP_DIR = os.path.join(OUTPUT_DIR, "sweeps")
DEFAULTS = dict(d_model=128, n_layers=2, num_bases=4, dropout=0.15, lr=1e-3)

# One axis varied at a time; every other value held at DEFAULTS.
AXES = {
    "d_model":  [64, 128, 256, 384],
    "n_layers": [1, 2, 3, 4],
    "num_bases": [2, 4, 8, 16],
    "dropout":  [0.0, 0.15, 0.3, 0.5],
    "lr":       [3e-4, 1e-3, 3e-3, 1e-2],
}


def _train_one(d, config, seed=42, verbose=False):
    """Train PatientConceptGNN with the given hyperparameters; return
    (best_val_mean_auroc_3y, best_epoch, best_state_dict)."""
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
        val_sids = np.array(d["splits"]["val"])
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

    return best_metric, best_epoch, best_state, time_edges, model, device


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

    print(f"\nScreening {len(configs)} configs (seed=42, val-only selection)...\n")
    for name, cfg in configs:
        t0 = time.time()
        best_metric, best_epoch, _, _, _, _ = _train_one(d, cfg, seed=42)
        dt = time.time() - t0
        print(f"  {name:16s} d_model={cfg['d_model']:4d} n_layers={cfg['n_layers']} "
              f"num_bases={cfg['num_bases']:2d} dropout={cfg['dropout']:.2f} lr={cfg['lr']:.4f}  "
              f"-> val_mean_AUROC@3y={best_metric:.4f} (best ep {best_epoch}, {dt:.1f}s)")
        rows.append(dict(name=name, **cfg, val_mean_auroc_3y=best_metric, best_epoch=best_epoch, seconds=dt))

    result = pd.DataFrame(rows)
    out_path = os.path.join(SWEEP_DIR, "patient_graph_gnn_sweep.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return result


def best_combined_config(result: pd.DataFrame) -> dict:
    """Per axis, pick the value with the highest val_mean_auroc_3y among rows
    that vary only that axis (baseline included in every axis's comparison
    pool); combine into one config."""
    best = dict(DEFAULTS)
    # int/float axes must stay native Python types, not numpy scalars --
    # PyG's RGCNConv does an isinstance(x, int) check internally that
    # silently fails (IndexError, not a clean TypeError) on np.int64/float64
    # pulled out of a pandas row via .loc/.iloc.
    axis_caster = {"d_model": int, "n_layers": int, "num_bases": int,
                   "dropout": float, "lr": float}
    baseline_score = result.loc[result["name"] == "baseline", "val_mean_auroc_3y"].iloc[0]
    print(f"\nBaseline val_mean_AUROC@3y = {baseline_score:.4f}")
    for axis in AXES:
        pool = result[result["name"].str.startswith(f"{axis}=") | (result["name"] == "baseline")]
        top = pool.loc[pool["val_mean_auroc_3y"].idxmax()]
        best[axis] = axis_caster[axis](top[axis])
        print(f"  best {axis}: {best[axis]} (val_mean_AUROC@3y={top['val_mean_auroc_3y']:.4f}, "
              f"{'improves' if top['val_mean_auroc_3y'] > baseline_score else 'no improvement'} over baseline)")
    return best


def run_final(best_cfg: dict) -> None:
    """Retrain the best-combined config across the standard 5 seeds and
    evaluate on TEST -- once, at the very end, same discipline as every
    other model in this study."""
    print(f"\nFinal best-combined config: {best_cfg}")
    out_dir = os.path.join(SWEEP_DIR, "patient_graph_gnn_best")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for seed in [42, 43, 44, 45, 46]:
        print(f"\n=== seed {seed} ===")
        d = _prepare_patient_graph_data()
        best_metric, best_epoch, best_state, time_edges, model, device = _train_one(d, best_cfg, seed=seed)
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
        test_metrics = _per_cause_auroc_at_horizons(cif_test, test_sids, d["labels_df"], time_edges, HORIZON_DAYS)
        test_metrics["seed"] = seed
        all_rows.append(test_metrics)
        print(f"  seed {seed}: best val_mean_AUROC@3y={best_metric:.4f} (epoch {best_epoch})")

    result = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(out_dir, "test_metrics.csv")
    result.to_csv(out_path, index=False)
    print(f"\n=== TUNED PATIENT-GRAPH GNN, 5-SEED TEST AUROC @ 3y ===")
    print(result[result.horizon_days == 1095].groupby("cause")["auroc"].agg(["mean", "std"]).round(4))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    screening = run_screening()
    best_cfg = best_combined_config(screening)
    run_final(best_cfg)
