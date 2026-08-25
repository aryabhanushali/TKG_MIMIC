"""Item 8: overlay validation curves across all four trained architectures.

Purely descriptive -- reuses the already-saved history.csv from each
canonical (seed 42) run, no retraining. The qualitative shape difference
(decline vs. plateau) is the point: it's visible evidence of a mechanistic
difference in what each architecture is doing, not just a number in a table.

Output: tkg_output/figures/fig22_training_stability.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, FIGURES_DIR
from src.tgn_survival import MIN_EPOCHS

RUNS = [
    ("tgn_survival", "Plain TKG-Transformer (sequence only)", "#1f77b4"),
    ("hetero_gnn_survival", "Concept graph, kept sequence (attempt 1, full)", "#d62728"),
    ("static_gnn_survival", "Concept graph, dropped sequence (attempt 1, static)", "#ff7f0e"),
    ("patient_gnn_survival", "Patient graph (attempt 2)", "#2ca02c"),
]


def run() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for d, label, color in RUNS:
        path = os.path.join(OUTPUT_DIR, d, "history.csv")
        if not os.path.exists(path):
            print(f"  skip (not found): {path}")
            continue
        df = pd.read_csv(path)
        ax.plot(df["epoch"], df["val_mean_auroc_3y"], "o-", label=label,
                color=color, linewidth=2, markersize=4)
    ax.axvline(MIN_EPOCHS, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"MIN_EPOCHS floor ({MIN_EPOCHS}, applies to all four models)")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation mean AUROC @ 3 years")
    ax.set_title("Validation trajectory by architecture (seed 42)\n"
                  "Sequence-based models peak early then decline; the patient graph does not",
                  fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig22_training_stability.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()
