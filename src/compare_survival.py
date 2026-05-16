"""Side-by-side: TGN-Survival vs Cox vs GBSA on per-cause AUROC at 1y/3y/5y."""
import os
import pandas as pd

from src.config import OUTPUT_DIR

CAUSES = ["MI", "Stroke", "HF", "AF", "PAD"]


def compare() -> None:
    parts = []
    cox_path = os.path.join(OUTPUT_DIR, "baselines_survival", "test_metrics.csv")
    tgn_path = os.path.join(OUTPUT_DIR, "tgn_survival", "test_metrics.csv")
    if os.path.exists(cox_path):
        parts.append(pd.read_csv(cox_path))
    if os.path.exists(tgn_path):
        tgn = pd.read_csv(tgn_path)
        tgn["model"] = "tgn_surv"
        parts.append(tgn)
    if not parts:
        print("No metrics files found.")
        return
    df = pd.concat(parts, ignore_index=True)
    print("\n=== TEST AUROC — per cause × horizon × model ===")
    for h in sorted(df["horizon_days"].unique()):
        sub = df[df["horizon_days"] == h]
        pv = sub.pivot(index="cause", columns="model", values="auroc")
        if not pv.empty:
            pv = pv.reindex(CAUSES).round(3)
            print(f"\nHorizon = {h} days:")
            print(pv.to_string())

    out = os.path.join(OUTPUT_DIR, "survival_comparison_test.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    compare()
