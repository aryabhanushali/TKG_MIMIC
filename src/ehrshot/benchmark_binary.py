"""TKG models on an EHRSHOT benchmark task (binary, e.g. new_acutemi).

Same features and same TKG-Transformer encoder as the survival pipeline, with
two changes so results compare directly with the published EHRSHOT
count / CLMBR baselines:
  * labels and prediction times come from EHRSHOT_MEDS/labels/<task>/, and
    splits from EHRSHOT_MEDS/metadata/subject_splits.parquet
    (train / tuning / held_out);
  * the head is a single logit trained with BCE.

For every label row the history is the TKG facts in
[prediction_time - 1825 d, prediction_time], built by the same extractors as
build_tkg_ehrshot.py (one shared code path). Models:
  lr / xgb : bag-of-codes + value summaries + static (baseline.py-style features)
  tkg_transformer : tgn_model.TKGTransformer with n_classes=1

Usage: python -m src.ehrshot.benchmark_binary [--task new_acutemi] [--epochs N]
Output: OUTPUT_DIR/benchmark/<task>/{test_metrics.csv, predictions_test.csv}
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader

from src.config import OUTPUT_DIR, SEED, PRE_INDEX_WINDOW_DAYS
from src.tgn_model import (TKGTransformer, PatientEventsDataset, collate, _set_seed,
                           MAX_SEQ_LEN, BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, PATIENCE)
from src.ehrshot import data_ehrshot as D
from src.tgn_survival import MIN_EPOCHS  # same checkpoint-eligibility floor as the survival model
from src.ehrshot.build_tkg_ehrshot import build_fact_pieces
from src.ehrshot.config_ehrshot import EHRSHOT_MEDS_DIR, EHRSHOT_ASSETS_DIR

SPLIT_MAP = {"train": "train", "tuning": "val", "held_out": "test"}
N_BOOT = 1000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_labels(task: str) -> pd.DataFrame:
    lab = pd.read_parquet(os.path.join(EHRSHOT_MEDS_DIR, "labels", task, "labels.parquet"))
    sp = pd.read_parquet(os.path.join(EHRSHOT_MEDS_DIR, "metadata", "subject_splits.parquet"))
    lab = lab.merge(sp, on="subject_id", how="inner")
    lab = lab[lab["subject_id"].isin(set(D.all_subjects()))]
    lab["split"] = lab["split"].map(SPLIT_MAP)
    lab["y"] = lab["boolean_value"].astype(int)
    lab["prediction_time"] = pd.to_datetime(lab["prediction_time"])
    lab = lab.sort_values(["subject_id", "prediction_time"]).reset_index(drop=True)
    lab["row"] = np.arange(len(lab))
    return lab[["row", "subject_id", "prediction_time", "y", "split"]]


def _label_events(lab: pd.DataFrame) -> pd.DataFrame:
    """One row per (label row, fact) with relative_days <= 0 to prediction_time."""
    w = lab.groupby("subject_id")["prediction_time"].agg(["min", "max"]).reset_index()
    windows = pd.DataFrame({"subject_id": w["subject_id"],
                            "window_start": w["min"] - pd.Timedelta(days=PRE_INDEX_WINDOW_DAYS),
                            "window_end": w["max"]})
    facts = pd.concat(build_fact_pieces(windows, inclusive_end=True), ignore_index=True)
    facts = facts.drop_duplicates(["subject_id", "relation", "concept_id", "timestamp_start"])
    ev = facts.merge(lab[["row", "subject_id", "prediction_time"]], on="subject_id")
    delta = ev["timestamp_start"] - ev["prediction_time"]
    ev = ev[(delta <= pd.Timedelta(0)) & (delta >= -pd.Timedelta(days=PRE_INDEX_WINDOW_DAYS))]
    ev["relative_days"] = delta.dt.total_seconds() / 86400.0
    return ev[["row", "relation", "concept_id", "relative_days", "value_num"]].reset_index(drop=True)


def _static(lab: pd.DataFrame) -> np.ndarray:
    demo = D.load_demographics().set_index("subject_id")
    age = (lab["prediction_time"] - lab["subject_id"].map(demo["birth"])).dt.days / 365.25
    female = (lab["subject_id"].map(demo["gender"]) == "F").astype(float)
    return np.stack([age.fillna(age.median()).to_numpy(), female.to_numpy()], 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Tabular features (bag-of-codes + value summaries + static), train-only vocab
# ---------------------------------------------------------------------------

def _tabular(lab, ev, static, train_rows):
    n = len(lab)
    tr_ev = ev[ev["row"].isin(train_rows)]
    vocab = sorted(tr_ev["concept_id"].unique())
    col = {c: j for j, c in enumerate(vocab)}
    cnt = ev[ev["concept_id"].isin(col)].groupby(["row", "concept_id"]).size().reset_index(name="n")
    X_bow = csr_matrix((cnt["n"].to_numpy(np.float32),
                        (cnt["row"].to_numpy(), cnt["concept_id"].map(col).to_numpy())),
                       shape=(n, len(vocab)))

    vc = sorted(tr_ev.loc[tr_ev["value_num"].notna(), "concept_id"].unique())
    vcol = {c: j for j, c in enumerate(vc)}
    ve = ev[ev["value_num"].notna() & ev["concept_id"].isin(vcol)].sort_values(["row", "relative_days"])
    g = ve.assign(xy=ve["relative_days"] * ve["value_num"], xx=ve["relative_days"] ** 2) \
          .groupby(["row", "concept_id"])
    a = g.agg(mean=("value_num", "mean"), max=("value_num", "max"), min=("value_num", "min"),
              last=("value_num", "last"), count=("value_num", "size"),
              sx=("relative_days", "sum"), sy=("value_num", "sum"), sxy=("xy", "sum"),
              sxx=("xx", "sum")).reset_index()
    den = a["count"] * a["sxx"] - a["sx"] ** 2
    a["slope"] = np.where(den.abs() > 1e-9, (a["count"] * a["sxy"] - a["sx"] * a["sy"]) / den.where(den.abs() > 1e-9, 1), 0.0)
    arr = np.zeros((n, len(vc) * 6), dtype=np.float32)
    base = a["concept_id"].map(vcol).to_numpy() * 6
    for k, s in enumerate(["mean", "max", "min", "last", "count", "slope"]):
        arr[a["row"].to_numpy(), base + k] = a[s].to_numpy(np.float32)
    tr_mask = np.isin(np.arange(n), train_rows)
    mu, sd = arr[tr_mask].mean(0), arr[tr_mask].std(0)
    sd[sd == 0] = 1.0
    arr = np.clip((arr - mu) / sd, -10, 10).astype(np.float32)
    smu, ssd = static[tr_mask].mean(0), static[tr_mask].std(0)
    ssd[ssd == 0] = 1.0
    X = hstack([X_bow, csr_matrix(arr), csr_matrix((static - smu) / ssd)]).tocsr()
    print(f"  tabular X: {X.shape} (bow {len(vocab):,}, values {arr.shape[1]})")
    return X


def _fit_lr(X, y, tr, va):
    sc = MaxAbsScaler().fit(X[tr])
    best = (None, -1, None)
    for C in (0.01, 0.1, 1.0):
        m = LogisticRegression(C=C, max_iter=3000, solver="liblinear", class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        s = roc_auc_score(y[va], m.predict_proba(sc.transform(X[va]))[:, 1])
        if s > best[1]:
            best = (m, s, C)
    print(f"  LR: best C={best[2]} (val AUROC {best[1]:.3f})")
    return lambda Z: best[0].predict_proba(sc.transform(Z))[:, 1]


def _fit_xgb(X, y, tr, va):
    m = xgb.XGBClassifier(objective="binary:logistic", n_estimators=400, max_depth=6,
                          learning_rate=0.08, subsample=0.85, colsample_bytree=0.7,
                          eval_metric="auc", early_stopping_rounds=20, tree_method="hist",
                          n_jobs=-1, random_state=SEED)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    print(f"  XGB: best_iteration={m.best_iteration}")
    return lambda Z: m.predict_proba(Z)[:, 1]


# ---------------------------------------------------------------------------
# TKG-Transformer with a binary head
# ---------------------------------------------------------------------------

def _sequences(lab, ev, static, train_rows):
    rel = sorted(ev["relation"].unique())
    rel_idx = {r: i for i, r in enumerate(rel)}
    tr_ev = ev[ev["row"].isin(train_rows)]
    remap = {c: i + 1 for i, c in enumerate(sorted(tr_ev["concept_id"].unique()))}
    st = tr_ev[tr_ev["value_num"].notna()].groupby("concept_id")["value_num"].agg(["mean", "std"])
    st["std"] = st["std"].fillna(1.0).replace(0, 1.0)
    ev = ev.assign(c=ev["concept_id"].map(remap).fillna(0).astype(np.int64),
                   e=ev["relation"].map(rel_idx).astype(np.int64))
    mu = ev["concept_id"].map(st["mean"])
    sd = ev["concept_id"].map(st["std"])
    present = ev["value_num"].notna() & mu.notna()
    ev["vn"] = np.where(present, ((ev["value_num"] - mu) / sd).clip(-10, 10), 0.0).astype(np.float32)
    ev["vp"] = present.astype(np.float32)
    ev = ev.sort_values(["row", "relative_days"])
    seqs = {r: (g["c"].to_numpy(), g["e"].to_numpy(), g["relative_days"].to_numpy(np.float32),
                g["vn"].to_numpy(np.float32), g["vp"].to_numpy(np.float32))
            for r, g in ev.groupby("row")}
    empty = (np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.float32),
             np.zeros(1, np.float32), np.zeros(1, np.float32))
    for r in lab["row"]:
        seqs.setdefault(r, empty)
    tr_mask = lab["row"].isin(train_rows).to_numpy()
    smu, ssd = static[tr_mask].mean(0), static[tr_mask].std(0)
    ssd[ssd == 0] = 1.0
    stat = {r: ((static[i] - smu) / ssd).astype(np.float32) for i, r in enumerate(lab["row"])}
    return seqs, stat, len(remap) + 1, len(rel)


def _predict(model, loader, device):
    model.eval()
    out, rows = [], []
    with torch.no_grad():
        for b in loader:
            logit = model(b["concept_idx"].to(device), b["edge_type_idx"].to(device), b["t"].to(device),
                          b["v_norm"].to(device), b["v_present"].to(device), b["static"].to(device),
                          b["mask"].to(device)).squeeze(-1)
            out.append(torch.sigmoid(logit).cpu().numpy())
            rows.append(b["sid"].numpy())
    return np.concatenate(rows), np.concatenate(out)


def _fit_transformer(lab, ev, static, train_rows, epochs):
    _set_seed(SEED)
    seqs, stat, n_concepts, n_edge = _sequences(lab, ev, static, train_rows)
    ylab = dict(zip(lab["row"], lab["y"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loader(split, shuffle):
        rows = lab.loc[lab["split"] == split, "row"].tolist()
        ds = PatientEventsDataset(rows, seqs, stat, ylab, max_len=MAX_SEQ_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate)

    tr_l, va_l, te_l = loader("train", True), loader("val", False), loader("test", False)
    model = TKGTransformer(n_concepts, n_edge, static.shape[1], n_classes=1).to(device)
    ytr = lab.loc[lab["split"] == "train", "y"]
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    yv = lab.set_index("row")["y"]
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for b in tr_l:
            logit = model(b["concept_idx"].to(device), b["edge_type_idx"].to(device), b["t"].to(device),
                          b["v_norm"].to(device), b["v_present"].to(device), b["static"].to(device),
                          b["mask"].to(device)).squeeze(-1)
            loss = lossf(logit, b["label"].float().to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(logit)
        sched.step()
        r, p = _predict(model, va_l, device)
        auc = roc_auc_score(yv.loc[r], p)
        print(f"  epoch {ep:2d} loss={tot / len(tr_l.dataset):.4f} val AUROC={auc:.4f}", flush=True)
        if ep < min(MIN_EPOCHS, epochs):
            continue   # too early to be eligible as "best" (see tgn_survival.MIN_EPOCHS)
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    r, p = _predict(model, te_l, device)
    print(f"  TKG-Transformer: best val AUROC {best:.4f}")
    return pd.Series(p, index=r)


# ---------------------------------------------------------------------------

def _metrics(y, p, model, rng):
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(roc_auc_score(y[i], p[i]))
    return dict(model=model, auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                auroc_lo=np.percentile(boots, 2.5), auroc_hi=np.percentile(boots, 97.5),
                n_pos=int(y.sum()), n=int(len(y)))


def _published(task: str) -> pd.DataFrame:
    p = os.path.join(EHRSHOT_ASSETS_DIR, "results", task, "all_results.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    r = pd.read_csv(p)
    r = r[(r["k"] == -1) & r["score"].isin(["auroc", "auprc"])]
    r = r.groupby(["model", "head", "score"])["value"].mean().unstack("score").reset_index()
    r["model"] = "published_" + r["model"] + "_" + r["head"]
    return r[["model", "auroc", "auprc"]]


def run(task: str, epochs: int) -> None:
    out_dir = os.path.join(OUTPUT_DIR, "benchmark", task + ("" if SEED == 42 else f"_seed{SEED}"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[1] Labels for {task}...")
    lab = _load_labels(task)
    print(lab.groupby("split")["y"].agg(["size", "sum"]).to_string())
    print("[2] Building per-label TKG histories...")
    ev = _label_events(lab)
    print(f"  label-events: {len(ev):,}; labels with no history: "
          f"{len(lab) - ev['row'].nunique():,}")
    static = _static(lab)
    train_rows = lab.loc[lab["split"] == "train", "row"].to_numpy()
    tr = (lab["split"] == "train").to_numpy()
    va = (lab["split"] == "val").to_numpy()
    te = (lab["split"] == "test").to_numpy()
    y = lab["y"].to_numpy()

    print("[3] Tabular baselines...")
    X = _tabular(lab, ev, static, train_rows)
    preds = pd.DataFrame({"row": lab.loc[te, "row"], "subject_id": lab.loc[te, "subject_id"],
                          "prediction_time": lab.loc[te, "prediction_time"], "y": y[te]})
    preds["lr"] = _fit_lr(X, y, tr, va)(X[te])
    preds["xgb"] = _fit_xgb(X, y, tr, va)(X[te])
    print("[4] TKG-Transformer (binary head)...")
    tp = _fit_transformer(lab, ev, static, train_rows, epochs)
    preds["tkg_transformer"] = preds["row"].map(tp).to_numpy()
    preds.to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False)

    rng = np.random.default_rng(SEED)
    res = pd.DataFrame([_metrics(preds["y"].to_numpy(), preds[m].to_numpy(), m, rng)
                        for m in ("lr", "xgb", "tkg_transformer")])
    res = pd.concat([res, _published(task)], ignore_index=True)
    res.insert(0, "task", task)
    res.to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
    print(f"\n=== {task}: held-out test ===")
    print(res.round(4).to_string(index=False))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="new_acutemi")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    run(a.task, a.epochs)
