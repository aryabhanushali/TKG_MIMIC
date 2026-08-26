"""Item 6b: does the patient-graph model's (attempt 2, the architecture that
actually works) implicit "which concepts drive this patient's prediction"
signal survive the same faithfulness check already run on the plain
TKG-Transformer (src/explain_gnn.py) and the failed concept-graph model
(src/ablations/explain_gnn_concept_graph.py)?

The patient-graph model trains full-batch over the whole shared graph -- one
forward pass produces every patient's prediction at once -- so the usual
per-patient GNNExplainer mask-optimization loop (100 full-graph forward
passes per patient) is impractical here. Instead this script uses a single
backward pass per patient: the gradient of that patient's predicted CIF for
their own true cause, with respect to the concept-embedding table, gives a
standard saliency score for every concept they are connected to (only paths
that actually reach that one patient's output contribute to this gradient,
so it is patient-specific despite the shared embedding table).

Saliency is then evaluated with the identical top-k/random-k KL-divergence
sufficiency + comprehensiveness test used by the other two models, but
implemented via literal edge editing: for one patient at a time, their own
patient<->concept edges are restricted to (sufficiency) or excluded from
(comprehensiveness) a chosen concept subset, holding every other patient's
and every concept-concept edge fixed, and the whole graph is re-run.

Output: tkg_output/explain/patient_graph_fidelity.csv
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import OUTPUT_DIR
from src.tgn_survival import CAUSES, NUM_CAUSES, NUM_TIME_BINS, _make_time_bins, _discretize
from src.ablations.patient_graph_gnn import PatientConceptGNN, _prepare_patient_graph_data
from src.fidelity_stats import summarize_fidelity, print_fidelity_summary

EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "explain")
MODEL_DIR = os.path.join(OUTPUT_DIR, "patient_gnn_survival")
MAX_PATIENTS_PER_CAUSE = 30
KEEP_FRAC = 0.20
HORIZON_FOR_SALIENCY = 1095  # 3y, the study's primary horizon
EPS = 1e-9  # matches src.explain_gnn's EPS so the two models' KL numbers are bit-comparable


def _probs_flat(model, static_all: torch.Tensor) -> torch.Tensor:
    logits = model(static_all).view(-1, NUM_CAUSES, NUM_TIME_BINS)
    flat = logits.reshape(logits.size(0), NUM_CAUSES * NUM_TIME_BINS)
    return F.softmax(flat, dim=-1)


def _cif_from_probs_flat(probs_flat: torch.Tensor) -> torch.Tensor:
    probs = probs_flat.reshape(-1, NUM_CAUSES, NUM_TIME_BINS)
    return torch.cumsum(probs, dim=-1)


def _own_edge_concept(edge_index: torch.Tensor, patient_node: int, n_concepts: int) -> torch.Tensor:
    """Length == n_edges. Value = the concept id at the other endpoint of any
    edge touching `patient_node`; -1 for edges that don't touch it at all."""
    src, dst = edge_index[0], edge_index[1]
    is_src = src == patient_node
    is_dst = dst == patient_node
    other = torch.where(is_src, dst, torch.where(is_dst, src, torch.full_like(src, -1)))
    other = torch.where(other < n_concepts, other, torch.full_like(other, -1))
    return other


def _masked_edges(edge_index, edge_type, own_concept, keep_concepts: torch.Tensor, keep: bool):
    """own_concept: (n_edges,) from _own_edge_concept. keep_concepts: concept
    ids to keep (sufficiency) or drop (comprehensiveness) among this
    patient's own edges. All non-owned edges pass through untouched."""
    is_own = own_concept >= 0
    in_set = torch.isin(own_concept, keep_concepts)
    if keep:
        row_mask = (~is_own) | (is_own & in_set)
    else:
        row_mask = (~is_own) | (is_own & ~in_set)
    return edge_index[:, row_mask], edge_type[row_mask]


def run() -> None:
    os.makedirs(EXPLAIN_DIR, exist_ok=True)
    print("Loading patient-graph data + model...")
    d = _prepare_patient_graph_data()
    n_concepts, n_patients = d["n_concepts"], d["n_patients"]
    labels_df, splits = d["labels_df"], d["splits"]
    pid_to_pos = d["pid_to_pos"]
    edge_index, edge_type = d["edge_index"], d["edge_type"]

    device = torch.device("cpu")
    model = PatientConceptGNN(
        n_concepts=n_concepts, n_patients=n_patients, n_static=d["n_static"],
        n_relations=d["n_relations"], edge_index=edge_index, edge_type=edge_type,
        n_classes=NUM_CAUSES * NUM_TIME_BINS,
    ).to(device)
    ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  loaded best_model.pt  device={device}")

    static_all = torch.as_tensor(np.asarray(d["static_arr"]), dtype=torch.float32, device=device)

    train_durations = labels_df.loc[
        labels_df["subject_id"].isin(set(splits["train"])), "time_to_event_days"
    ].to_numpy(dtype=np.float32)
    time_edges = _make_time_bins(train_durations, NUM_TIME_BINS)
    horizon_bin = int(_discretize(np.array([HORIZON_FOR_SALIENCY]), time_edges)[0])
    print(f"  horizon {HORIZON_FOR_SALIENCY}d -> time bin {horizon_bin}")

    sid_to_label = dict(zip(labels_df["subject_id"], labels_df["endpoint_type"]))
    cause_to_idx = {c: i for i, c in enumerate(CAUSES)}
    selected = []
    for cause in CAUSES:
        c_sids = [s for s in splits["test"] if sid_to_label.get(s) == cause and s in pid_to_pos]
        selected.extend(c_sids[:MAX_PATIENTS_PER_CAUSE])
    print(f"  explaining {len(selected)} true-positive test patients (<= {MAX_PATIENTS_PER_CAUSE} per cause)")

    orig_edge_index, orig_edge_type = model.edge_index, model.edge_type
    with torch.no_grad():
        probs_full = _probs_flat(model, static_all)

    fidelity_rows = []
    for n_done, sid in enumerate(selected, 1):
        pos = pid_to_pos[sid]
        patient_node = n_concepts + pos
        cause_idx = cause_to_idx[sid_to_label[sid]]

        own_concept = _own_edge_concept(orig_edge_index, patient_node, n_concepts)
        neighbor_concepts = own_concept[own_concept >= 0].unique()
        n = int(neighbor_concepts.numel())
        if n < 2:
            continue

        model.zero_grad(set_to_none=True)
        probs = _probs_flat(model, static_all)
        cif = _cif_from_probs_flat(probs)
        target = cif[pos, cause_idx, horizon_bin]
        target.backward()
        grad = model.concept_emb.weight.grad
        saliency = grad[neighbor_concepts].norm(dim=1).detach().numpy()

        k = max(int(round(n * KEEP_FRAC)), 1)
        order = np.argsort(-saliency)
        top_concepts = neighbor_concepts[order[:k]]
        rng = np.random.default_rng(int(sid))
        rand_concepts = neighbor_concepts[torch.as_tensor(
            rng.choice(n, size=k, replace=False), dtype=torch.long)]

        p_orig = probs_full[pos].detach()
        log_orig = torch.log(p_orig + EPS)

        def _kl(concept_set: torch.Tensor, keep: bool) -> float:
            ei, et = _masked_edges(orig_edge_index, orig_edge_type, own_concept, concept_set, keep)
            model.edge_index, model.edge_type = ei, et
            try:
                with torch.no_grad():
                    p = _probs_flat(model, static_all)[pos]
            finally:
                # Must always restore the shared graph, even if the forward
                # pass raises -- otherwise every remaining patient in the
                # loop would silently run on a corrupted, single-patient-
                # restricted edge set.
                model.edge_index, model.edge_type = orig_edge_index, orig_edge_type
            return float((p_orig * (log_orig - torch.log(p + EPS))).sum())

        fidelity_rows.append({
            "subject_id": int(sid), "cause": sid_to_label[sid], "n_concepts": n,
            "keep_frac": KEEP_FRAC,
            "kl_keep_top": _kl(top_concepts, True),
            "kl_keep_random": _kl(rand_concepts, True),
            "kl_drop_top": _kl(top_concepts, False),
            "kl_drop_random": _kl(rand_concepts, False),
        })
        if n_done % 25 == 0 or n_done == len(selected):
            print(f"  explained {n_done}/{len(selected)} patients")

    fidelity = pd.DataFrame(fidelity_rows)
    fid_path = os.path.join(EXPLAIN_DIR, "patient_graph_fidelity.csv")
    fidelity.to_csv(fid_path, index=False)
    pct = int(KEEP_FRAC * 100)
    print("\nFidelity (patient-graph model, mean over explained patients, top-k vs random-k):")
    print(f"  sufficiency  KL(keep top-{pct}%)  = {fidelity['kl_keep_top'].mean():.4f}"
          f"  vs random = {fidelity['kl_keep_random'].mean():.4f}  (top should be LOWER)")
    print(f"  comprehens.  KL(drop top-{pct}%)  = {fidelity['kl_drop_top'].mean():.4f}"
          f"  vs random = {fidelity['kl_drop_random'].mean():.4f}  (top should be HIGHER)")

    # The raw KL magnitudes here are ~1000x smaller than the other two
    # models' (one patient's own edges are a tiny fraction of a multi-million-
    # edge shared graph), so a mean comparison alone risks looking "decisive"
    # from a handful of large-KL patients even if a typical patient shows no
    # reliable effect. Win-rate + Wilcoxon (not just the paired t-test) is the
    # right way to check that -- see src/fidelity_stats.py.
    fidelity_summary = summarize_fidelity(fidelity)
    fidelity_summary.to_csv(os.path.join(EXPLAIN_DIR, "patient_graph_fidelity_stats.csv"))
    print_fidelity_summary(fidelity_summary, "Patient-graph model (attempt 2)")

    print(f"\nSaved: {fid_path}")


if __name__ == "__main__":
    run()
