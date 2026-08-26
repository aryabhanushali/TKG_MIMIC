"""Follow-up to patient_graph_gnn_sweep.py: that script's coordinate-wise
sweep found that combining each axis's individually-best value produced a
WORSE, less stable model than the defaults (val AUROC collapsed on 2 of 5
final seeds) -- evidence that these hyperparameters interact, so a greedy
per-axis combination doesn't work. This script runs a bounded RANDOM search
over the JOINT space instead (still not exhaustive -- true Bayesian
optimization or a full grid would cost far more compute than this project's
hardware supports in reasonable time -- but a genuine step beyond "vary one
knob at a time").

Search space capped below the most expensive extremes explored in the
coordinate-wise sweep (d_model<=256, n_layers<=3) specifically because those
combinations were the slowest and, combined, were the ones that destabilized
training -- deliberately searching the region actually reachable in this
project's compute budget, not the theoretical maximum.

Selection: validation mean AUROC @ 3y, seed 42 only for screening (matches
every other selection rule in this study). The single best joint config is
then confirmed across all 5 standard seeds -- test data touched once, at
the very end, for that one config only.

Output: tkg_output/sweeps/patient_graph_gnn_joint_sweep.csv
        tkg_output/sweeps/patient_graph_gnn_joint_best/test_metrics.csv
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import OUTPUT_DIR
from src.ablations.patient_graph_gnn import _prepare_patient_graph_data
from src.ablations.patient_graph_gnn_sweep import _train_one, DEFAULTS

SWEEP_DIR = os.path.join(OUTPUT_DIR, "sweeps")

# Capped below the most expensive coordinate-wise extremes (d_model=384,
# n_layers=4) -- those were both the slowest to train AND, combined, the
# ones that destabilized the greedy-combined config.
SPACE = dict(
    d_model=[64, 128, 256],
    n_layers=[1, 2, 3],
    num_bases=[2, 4, 8, 16],
    dropout=[0.0, 0.15, 0.3],
    lr=[3e-4, 1e-3, 3e-3],
)
N_RANDOM_CONFIGS = 12


def sample_configs(n: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    configs = [("baseline", dict(DEFAULTS))]
    seen = {tuple(sorted(DEFAULTS.items()))}
    while len(configs) < n + 1:
        cfg = {}
        for k, v in SPACE.items():
            pick = v[rng.integers(0, len(v))]
            cfg[k] = int(pick) if isinstance(pick, (int, np.integer)) else float(pick)
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append((f"joint_{len(configs)}", cfg))
    return configs


def run_screening() -> pd.DataFrame:
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print("Loading graph data once (shared across all sweep configs)...")
    d = _prepare_patient_graph_data()

    configs = sample_configs(N_RANDOM_CONFIGS, seed=0)
    print(f"\nScreening {len(configs)} joint-random configs (seed=42, val-only selection)...\n")
    rows = []
    for name, cfg in configs:
        t0 = time.time()
        best_metric, best_epoch, _, _, _, _ = _train_one(d, cfg, seed=42)
        dt = time.time() - t0
        print(f"  {name:10s} d_model={cfg['d_model']:4d} n_layers={cfg['n_layers']} "
              f"num_bases={cfg['num_bases']:2d} dropout={cfg['dropout']:.2f} lr={cfg['lr']:.4f}  "
              f"-> val_mean_AUROC@3y={best_metric:.4f} (best ep {best_epoch}, {dt:.1f}s)")
        rows.append(dict(name=name, **cfg, val_mean_auroc_3y=best_metric, best_epoch=best_epoch, seconds=dt))

    result = pd.DataFrame(rows)
    out_path = os.path.join(SWEEP_DIR, "patient_graph_gnn_joint_sweep.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return result


def run_final(best_cfg: dict) -> None:
    print(f"\nBest joint config: {best_cfg}")
    out_dir = os.path.join(SWEEP_DIR, "patient_graph_gnn_joint_best")
    os.makedirs(out_dir, exist_ok=True)

    from src.tgn_survival import CAUSES, NUM_CAUSES, NUM_TIME_BINS, HORIZON_DAYS, _per_cause_auroc_at_horizons

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
    print(f"\n=== JOINT-TUNED PATIENT-GRAPH GNN, 5-SEED TEST AUROC @ 3y ===")
    print(result[result.horizon_days == 1095].groupby("cause")["auroc"].agg(["mean", "std"]).round(4))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    screening = run_screening()
    best_row = screening.loc[screening["val_mean_auroc_3y"].idxmax()]
    best_cfg = {k: (int(best_row[k]) if k in ("d_model", "n_layers", "num_bases") else float(best_row[k]))
                for k in ["d_model", "n_layers", "num_bases", "dropout", "lr"]}
    baseline_score = screening.loc[screening["name"] == "baseline", "val_mean_auroc_3y"].iloc[0]
    print(f"\nBaseline val_mean_AUROC@3y = {baseline_score:.4f}")
    print(f"Best joint-random val_mean_AUROC@3y = {best_row['val_mean_auroc_3y']:.4f} ({best_row['name']})")
    run_final(best_cfg)
