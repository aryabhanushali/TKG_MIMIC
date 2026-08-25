"""Item 4: does the R-GCN concept-enrichment step (attempt 1, the design that
failed) oversmooth concept embeddings toward their shared ontology hub nodes?

Loads the already-trained concept-graph checkpoint (no retraining) and
compares concept embeddings before vs. after message-passing:
  - effective rank (participation ratio of the singular-value spectrum):
    lower post-message-passing rank means embeddings collapsed into fewer
    effective directions -- the standard, quantitative GNN oversmoothing
    signature.
  - mean pairwise cosine similarity: rising post-message-passing similarity
    means concepts that were previously distinguishable became more alike.
  - hub-degree stratified similarity: is the collapse worse for concepts
    connected to high-degree (chapter-level) hub nodes specifically, which
    is the mechanistic claim (hub-node bottleneck), not just "message
    passing smooths everything a little"?

Output: tkg_output/stats/oversmoothing_diagnostic.csv
        tkg_output/figures/fig23_oversmoothing.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import torch

# Apple Accelerate's BLAS backend spuriously raises "divide by zero" /
# "overflow" / "invalid value" RuntimeWarnings on some float32 matmuls with
# no actual NaN/Inf in the output (verified directly against this exact
# checkpoint's embeddings and against synthetic data) -- suppress just this
# known-benign class of warning rather than the ones that would flag a real
# problem elsewhere in the script.
warnings.filterwarnings("ignore", message=".*encountered in matmul")
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, FIGURES_DIR
from src.ablations.hetero_gnn import N_RGCN_LAYERS

STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
CKPT_PATH = os.path.join(OUTPUT_DIR, "hetero_gnn_survival", "best_model.pt")
N_SAMPLE = 3000  # pairwise similarity on a random sample, for tractability


def _effective_rank(X: np.ndarray) -> float:
    """Participation ratio of the singular-value spectrum: (sum(s^2))^2 /
    sum(s^4). Equals the true rank for a spectrum with equal singular
    values, and collapses toward 1 as one direction dominates -- a
    continuous, standard measure of representational collapse."""
    s = np.linalg.svd(X, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def run() -> None:
    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading concept-graph checkpoint (no retraining)...")
    assert N_RGCN_LAYERS == 1, (
        f"hetero_gnn.N_RGCN_LAYERS={N_RGCN_LAYERS}, but this script only "
        "reconstructs a single RGCNConv layer (rgcn.0.*) from the checkpoint "
        "-- it would silently compare pre- vs. after-only-the-first-layer "
        "embeddings instead of the real multi-layer stack. Update this "
        "script to reconstruct all layers before removing this assertion."
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    node_emb = sd["encoder.concept_table.node_emb.weight"]
    edge_index = sd["encoder.concept_table.edge_index"]
    edge_type = sd["encoder.concept_table.edge_type"]
    n_concepts, d_model = node_emb.shape
    n_relations = int(edge_type.max().item()) + 1
    print(f"  {n_concepts:,} concept nodes, {edge_index.shape[1]:,} edges, "
          f"{n_relations} relations, d_model={d_model}")

    rgcn = RGCNConv(d_model, d_model, num_relations=n_relations, num_bases=4)
    rgcn.load_state_dict({
        "weight": sd["encoder.concept_table.rgcn.0.weight"],
        "comp": sd["encoder.concept_table.rgcn.0.comp"],
        "root": sd["encoder.concept_table.rgcn.0.root"],
        "bias": sd["encoder.concept_table.rgcn.0.bias"],
    })
    norm = nn.LayerNorm(d_model)
    norm.load_state_dict({
        "weight": sd["encoder.concept_table.norm.weight"],
        "bias": sd["encoder.concept_table.norm.bias"],
    })

    with torch.no_grad():
        x0 = node_emb
        x1 = F.gelu(rgcn(x0, edge_index, edge_type))
        x_final = norm(x1 + x0)   # matches OntologyConceptTable.forward()'s residual

    pre = x0.numpy()
    post = x_final.numpy()

    print("\nEffective rank (participation ratio of singular-value spectrum):")
    er_pre, er_post = _effective_rank(pre), _effective_rank(post)
    print(f"  pre-message-passing:  {er_pre:.1f}  (max possible = {d_model})")
    print(f"  post-message-passing: {er_post:.1f}")
    print(f"  collapse ratio (post/pre): {er_post / er_pre:.3f}  "
          f"(1.0 = no change, <1.0 = collapse toward fewer directions)")

    rng = np.random.default_rng(42)
    idx = rng.choice(n_concepts, size=min(N_SAMPLE, n_concepts), replace=False)
    pre_n = pre[idx] / (np.linalg.norm(pre[idx], axis=1, keepdims=True) + 1e-9)
    post_n = post[idx] / (np.linalg.norm(post[idx], axis=1, keepdims=True) + 1e-9)
    sim_pre = pre_n @ pre_n.T
    sim_post = post_n @ post_n.T
    triu = np.triu_indices(len(idx), k=1)
    mean_sim_pre = float(sim_pre[triu].mean())
    mean_sim_post = float(sim_post[triu].mean())
    print(f"\nMean pairwise cosine similarity ({len(idx):,}-concept sample):")
    print(f"  pre-message-passing:  {mean_sim_pre:.4f}")
    print(f"  post-message-passing: {mean_sim_post:.4f}")

    # Hub-degree stratified: does similarity rise more for high-degree-neighbor concepts?
    deg = np.zeros(n_concepts)
    src = edge_index[0].numpy()
    np.add.at(deg, src, 1)
    deg_sample = deg[idx]
    hi_mask = deg_sample >= np.median(deg_sample[deg_sample > 0]) if (deg_sample > 0).any() else np.zeros_like(deg_sample, dtype=bool)

    def _mean_sim_for(mask, sim):
        sub_idx = np.where(mask)[0]
        if len(sub_idx) < 2:
            return np.nan
        sub = sim[np.ix_(sub_idx, sub_idx)]
        tri = np.triu_indices(len(sub_idx), k=1)
        return float(sub[tri].mean())

    hi_pre, hi_post = _mean_sim_for(hi_mask, sim_pre), _mean_sim_for(hi_mask, sim_post)
    lo_pre, lo_post = _mean_sim_for(~hi_mask, sim_pre), _mean_sim_for(~hi_mask, sim_post)
    print(f"\nHub-degree stratified (median-split by number of ontology edges):")
    print(f"  high-degree concepts:  pre={hi_pre:.4f} -> post={hi_post:.4f}  "
          f"(delta={hi_post - hi_pre:+.4f})")
    print(f"  low-degree concepts:   pre={lo_pre:.4f} -> post={lo_post:.4f}  "
          f"(delta={lo_post - lo_pre:+.4f})")

    result = pd.DataFrame([{
        "effective_rank_pre": er_pre, "effective_rank_post": er_post,
        "collapse_ratio": er_post / er_pre,
        "mean_cosine_sim_pre": mean_sim_pre, "mean_cosine_sim_post": mean_sim_post,
        "high_degree_sim_pre": hi_pre, "high_degree_sim_post": hi_post,
        "low_degree_sim_pre": lo_pre, "low_degree_sim_post": lo_post,
    }])
    out_path = os.path.join(STATS_DIR, "oversmoothing_diagnostic.csv")
    result.to_csv(out_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(["pre", "post"], [er_pre, er_post], color=["#1f77b4", "#d62728"])
    axes[0].set_ylabel("Effective rank (participation ratio)")
    axes[0].set_title("Representational collapse", fontweight="bold")
    bars = axes[1].bar(["low-degree\npre", "low-degree\npost", "high-degree\npre", "high-degree\npost"],
                        [lo_pre, lo_post, hi_pre, hi_post],
                        color=["#1f77b4", "#1f77b4", "#d62728", "#d62728"])
    for b, a in zip(bars, [0.5, 1.0, 0.5, 1.0]):
        b.set_alpha(a)
    axes[1].set_ylabel("Mean pairwise cosine similarity")
    axes[1].set_title("Hub-degree stratified similarity increase", fontweight="bold")
    fig.suptitle("Oversmoothing diagnostic: concept-graph (attempt 1) embeddings", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = os.path.join(FIGURES_DIR, "fig23_oversmoothing.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved:\n  {out_path}\n  {fig_path}")


if __name__ == "__main__":
    run()
