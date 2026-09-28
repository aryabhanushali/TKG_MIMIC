"""Hyperparameter tuning for the TKG-Transformer: a bounded RANDOM search
over the joint hyperparameter space (d_model, n_heads, n_layers, dropout,
lr), in the same spirit as patient_graph_gnn_joint_sweep.py.

Unlike the patient-graph GNN, the TKG-Transformer has never had a
hyperparameter search of any kind -- its architecture (128-dim, 4 heads,
2 layers, dropout 0.15, lr 1e-3) is just the value tgn_model.py shipped
with. This script checks whether that choice is actually good, using the
same discipline as every other search in this project: screen candidates
on the training-cohort-only-derived split's validation set, select by
mean AUROC at the 3-year horizon (matching tgn_survival.py's own
early-stopping metric), and never touch the test set during screening.

Search space is intentionally small relative to the patient-graph GNN's
(6 random configs instead of 12): a full TKG-Transformer run costs
20-50 minutes versus ~15-25 seconds for the patient-graph GNN, so a
12-config screen here would cost hours more. d_model/n_heads pairs are
restricted to values where d_model is divisible by n_heads (a hard
requirement of multi-head attention).

The MIN_EPOCHS=15 floor and PATIENCE=6 early-stopping rule from
tgn_survival.py are preserved unchanged for every candidate -- this
search tunes architecture/optimization hyperparameters, not the
checkpoint-selection discipline documented in the paper.

This script only screens configs at seed 42 and reports the result; it
does not retrain the winner across all 5 seeds. That confirmation step
(mirroring patient_graph_gnn_joint_sweep.py's run_final) is deliberately
left as a separate, explicit follow-up once a promising config is known,
given the cost of retraining a TKG-Transformer 5 times over.

Output: tkg_output/sweeps/tgn_survival_joint_sweep.csv
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import OUTPUT_DIR
from src.tgn_model import (
    PatientEventsDataset, collate, MAX_SEQ_LEN,
    BATCH_SIZE, WEIGHT_DECAY, EPOCHS, PATIENCE, _set_seed, _prepare_data,
)
from src.tgn_survival import (
    TKGSurvivalNet, NUM_CAUSES, NUM_TIME_BINS, HORIZON_DAYS, MIN_EPOCHS,
    _make_time_bins, _prepare_survival_targets, _evaluate_survival,
    _per_cause_auroc_at_horizons, _deephit_nll_per_sample,
)

SWEEP_DIR = os.path.join(OUTPUT_DIR, "sweeps")
DEFAULTS = dict(d_model=128, n_heads=4, n_layers=2, dropout=0.15, lr=1e-3)

# d_model choices deliberately kept divisible by every n_heads choice.
SPACE = dict(
    d_model=[64, 128, 256],
    n_heads=[2, 4, 8],
    n_layers=[1, 2, 3],
    dropout=[0.0, 0.15, 0.3],
    lr=[3e-4, 1e-3, 3e-3],
)
N_RANDOM_CONFIGS = 6


def sample_configs(n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    configs = [("baseline", dict(DEFAULTS))]
    seen = {tuple(sorted(DEFAULTS.items()))}
    attempts = 0
    while len(configs) < n + 1 and attempts < n * 20:
        attempts += 1
        cfg = {}
        for k, v in SPACE.items():
            pick = v[rng.integers(0, len(v))]
            cfg[k] = int(pick) if isinstance(pick, (int, np.integer)) else float(pick)
        if cfg["d_model"] % cfg["n_heads"] != 0:
            continue
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append((f"joint_{len(configs)}", cfg))
    return configs


def _train_one(data, config: dict, seed: int = 42):
    """Train one TKGSurvivalNet with `config`; return
    (best_val_mean_auroc_3y, best_epoch, best_state, time_edges)."""
    _set_seed(seed)
    (events_by_sid, static_by_sid, _, splits,
     n_concepts, n_edge_types, n_static, labels_df) = data

    train_sids = splits["train"]
    train_durations = labels_df.loc[
        labels_df["subject_id"].isin(train_sids), "time_to_event_days"
    ].to_numpy(dtype=np.float32)
    time_edges = _make_time_bins(train_durations, NUM_TIME_BINS)

    survival_targets = _prepare_survival_targets(labels_df, time_edges)
    label_by_sid = {sid: int(t[1]) for sid, t in survival_targets.items()}
    duration_by_sid = {sid: int(t[0]) for sid, t in survival_targets.items()}

    def _make_loader(sids, shuffle):
        ds = PatientEventsDataset(sids, events_by_sid, static_by_sid,
                                    label_by_sid, max_len=MAX_SEQ_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                            collate_fn=collate, num_workers=0)

    train_loader = _make_loader(splits["train"], shuffle=True)
    val_loader = _make_loader(splits["val"], shuffle=False)

    # Forced to CPU: PyTorch's MPS backend (MPSGraph) compiles and caches a
    # distinct graph file per unique input shape, and collate() in
    # tgn_model.py pads each batch to that batch's own max length rather
    # than a fixed MAX_SEQ_LEN -- so a sweep with many varied configs hits
    # many distinct shapes and can fill the disk with mpsgraph cache files
    # (this is what caused the ENOSPC crash mid-sweep). CPU has no such
    # cache and is the safe choice for repeated sweep runs.
    device = torch.device("cpu")

    model = TKGSurvivalNet(
        n_concepts=n_concepts, n_edge_types=n_edge_types, n_static=n_static,
        num_causes=NUM_CAUSES, num_time_bins=NUM_TIME_BINS,
        d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], dropout=config["dropout"],
    ).to(device)

    counts = np.zeros(NUM_CAUSES + 1)
    for sid in splits["train"]:
        counts[label_by_sid[sid]] += 1
    weights = np.ones_like(counts)
    weights[1:] = (counts.sum() / (NUM_CAUSES * counts[1:].clip(min=1)))
    weights = weights / weights.mean()
    sample_weight_by_event = torch.tensor(weights, dtype=torch.float32, device=device)

    def weighted_deephit_nll(logits, dur_idx, evt_idx):
        per_sample = _deephit_nll_per_sample(logits, dur_idx, evt_idx)
        w = sample_weight_by_event[evt_idx]
        return (per_sample * w).mean()

    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_metric, best_epoch, no_improve, best_state = -1.0, -1, 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            for k in ("concept_idx", "edge_type_idx", "t", "v_norm", "v_present", "static", "mask"):
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
        scheduler.step()

        cif_val, sids_val = _evaluate_survival(model, val_loader, device)
        val_metrics = _per_cause_auroc_at_horizons(cif_val, sids_val, labels_df, time_edges, HORIZON_DAYS)
        mean3y = float(val_metrics[val_metrics["horizon_days"] == 1095]["auroc"].mean(skipna=True))

        if epoch < MIN_EPOCHS:
            continue
        if mean3y > best_metric:
            best_metric, best_epoch, no_improve = mean3y, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    return best_metric, best_epoch, best_state, time_edges


def run_screening() -> pd.DataFrame:
    os.makedirs(SWEEP_DIR, exist_ok=True)
    out_path = os.path.join(SWEEP_DIR, "tgn_survival_joint_sweep.csv")

    # Resume support: skip configs already completed in a prior run (e.g.
    # after the ENOSPC crash), rather than retraining them from scratch.
    rows = []
    done_names = set()
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        rows = prior.to_dict("records")
        done_names = set(prior["name"])
        print(f"Resuming: {len(done_names)} config(s) already completed ({sorted(done_names)}).")

    print("Loading data once (shared across all sweep configs)...")
    data = _prepare_data()

    configs = sample_configs(N_RANDOM_CONFIGS, seed=0)
    configs = [(name, cfg) for name, cfg in configs if name not in done_names]
    print(f"\nScreening {len(configs)} remaining joint-random config(s) (seed=42, val-only selection, "
          f"mean AUROC @ 3y across 5 causes)...\n")
    for name, cfg in configs:
        t0 = time.time()
        best_metric, best_epoch, _, _ = _train_one(data, cfg, seed=42)
        dt = time.time() - t0
        print(f"  {name:10s} d_model={cfg['d_model']:4d} n_heads={cfg['n_heads']} "
              f"n_layers={cfg['n_layers']} dropout={cfg['dropout']:.2f} lr={cfg['lr']:.4f}  "
              f"-> val_mean_AUROC@3y={best_metric:.4f} (best ep {best_epoch}, {dt/60:.1f} min)", flush=True)
        rows.append(dict(name=name, **cfg, val_mean_auroc_3y=best_metric,
                          best_epoch=best_epoch, seconds=dt))
        pd.DataFrame(rows).to_csv(os.path.join(SWEEP_DIR, "tgn_survival_joint_sweep.csv"), index=False)

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    baseline_score = result.loc[result["name"] == "baseline", "val_mean_auroc_3y"].iloc[0]
    best_row = result.loc[result["val_mean_auroc_3y"].idxmax()]
    print(f"\nBaseline (current defaults) val_mean_AUROC@3y = {baseline_score:.4f}")
    print(f"Best candidate ({best_row['name']}) val_mean_AUROC@3y = {best_row['val_mean_auroc_3y']:.4f}")
    return result


def run_final(best_cfg: dict, seeds=(42, 43, 44, 45, 46)) -> None:
    """Retrain `best_cfg` across all 5 standard seeds and evaluate on the
    test set once per seed -- the confirmation step patient_graph_gnn_joint_
    sweep.py's run_final() already does for the patient-graph GNN. Test data
    is touched exactly once per seed, only for this one already-selected
    config; no other config from the screening phase is evaluated on test.

    Resume-safe: skips any seed already present in the output CSV.
    """
    out_dir = os.path.join(SWEEP_DIR, "tgn_joint_best")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_metrics.csv")

    done_seeds = set()
    all_rows = []
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        all_rows = [prior]
        done_seeds = set(prior["seed"].unique().tolist())
        print(f"Resuming: seed(s) already completed: {sorted(done_seeds)}")

    print(f"\nConfirming best config across seeds: {best_cfg}\n")
    print("Loading data once (shared across all seeds)...")
    data = _prepare_data()
    (events_by_sid, static_by_sid, _, splits,
     n_concepts, n_edge_types, n_static, labels_df) = data

    for seed in seeds:
        if seed in done_seeds:
            print(f"  seed {seed}: already done, skipping")
            continue
        print(f"\n=== seed {seed} ===")
        t0 = time.time()
        best_metric, best_epoch, best_state, time_edges = _train_one(data, best_cfg, seed=seed)
        dt = time.time() - t0
        print(f"  best val_mean_AUROC@3y={best_metric:.4f} (epoch {best_epoch}, {dt/60:.1f} min)")

        device = torch.device("cpu")
        model = TKGSurvivalNet(
            n_concepts=n_concepts, n_edge_types=n_edge_types, n_static=n_static,
            num_causes=NUM_CAUSES, num_time_bins=NUM_TIME_BINS,
            d_model=best_cfg["d_model"], n_heads=best_cfg["n_heads"],
            n_layers=best_cfg["n_layers"], dropout=best_cfg["dropout"],
        ).to(device)
        if best_state is not None:
            model.load_state_dict(best_state)

        survival_targets = _prepare_survival_targets(labels_df, time_edges)
        label_by_sid = {sid: int(t[1]) for sid, t in survival_targets.items()}
        test_loader = DataLoader(
            PatientEventsDataset(splits["test"], events_by_sid, static_by_sid,
                                  label_by_sid, max_len=MAX_SEQ_LEN),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0,
        )
        cif_test, sids_test = _evaluate_survival(model, test_loader, device)
        test_metrics = _per_cause_auroc_at_horizons(cif_test, sids_test, labels_df, time_edges, HORIZON_DAYS)
        test_metrics["seed"] = seed
        test_metrics["best_epoch"] = best_epoch
        test_metrics["val_mean_auroc_3y"] = best_metric
        all_rows.append(test_metrics)

        pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)
        print(f"  seed {seed} test AUROC (3y): "
              f"{test_metrics[test_metrics.horizon_days == 1095][['cause', 'auroc']].to_string(index=False)}")

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(out_path, index=False)
    print(f"\n=== TUNED TKG-TRANSFORMER (joint_3 config), 5-SEED TEST AUROC ===")
    print(result.pivot_table(index="cause", columns="horizon_days", values="auroc", aggfunc=["mean", "std"]).round(4))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    BEST_CFG = dict(d_model=128, n_heads=8, n_layers=2, dropout=0.15, lr=3e-3)
    run_final(BEST_CFG)
