"""Turn this dataset's own validation sweeps into hparams_survival.json.

baselines_survival.py hard-codes COX_ALPHA / XGB_CONFIG tuned on MIMIC. For a
different dataset those values are not transferable (different feature count,
cohort size, event rates), so after running
  python -m src.ablations.cox_regularization_sweep
  python -m src.ablations.xgb_survival_sweep
on the EHRSHOT output dir, this script writes the validation-selected values
to OUTPUT_DIR/hparams_survival.json. baselines_survival.py picks that file up
automatically. Selection is validation-only in both sweeps; test is untouched.
"""
import json
import os

import pandas as pd

from src.config import OUTPUT_DIR
from src.ablations.xgb_survival_sweep import best_combined_config


def run() -> None:
    cox = pd.read_csv(os.path.join(OUTPUT_DIR, "stats", "cox_regularization_sweep.csv"))
    cox_alpha = {c: float(a) for c, a in
                 cox.groupby("cause")["tuned_alpha"].first().items()}
    screening = pd.read_csv(os.path.join(OUTPUT_DIR, "sweeps", "xgb_survival_sweep.csv"))
    best = best_combined_config(screening)
    xgb_cfg = {k: (int(v) if k in ("n_estimators", "max_depth") else float(v))
               for k, v in best.items()}
    out = {"cox_alpha": cox_alpha, "xgb_config": xgb_cfg,
           "source": "validation-selected by cox_regularization_sweep.py / xgb_survival_sweep.py"}
    path = os.path.join(OUTPUT_DIR, "hparams_survival.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    run()
