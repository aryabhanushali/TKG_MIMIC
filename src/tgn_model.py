"""Transformer over patient pre-index event sequences, using a TGAT-style
(Bochner/time2vec) time encoding for each event's relative timestamp -- the
architecture itself is a plain sequence Transformer with attention pooling,
not TGAT's temporal-neighbor message-passing."""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, f1_score, accuracy_score, log_loss,
)

from src.config import OUTPUT_DIR, FIGURES_DIR, SEED, read_events_table

MODELING_DIR = os.path.join(OUTPUT_DIR, "modeling")
MODEL_DIR = os.path.join(OUTPUT_DIR, "tgn")
ENDPOINT_ORDER = ["MI", "Stroke", "HF", "AF", "PAD", "censored"]
EP_TO_IDX = {ep: i for i, ep in enumerate(ENDPOINT_ORDER)}

MAX_SEQ_LEN = 256
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.15
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 30
PATIENCE = 6


class TimeEncoder(nn.Module):
    """Bochner time encoding (Xu et al. 2020, TGAT)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        freqs = 1.0 / (10.0 ** (torch.arange(dim).float() / dim))
        self.basis_freq = nn.Parameter(freqs)
        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out = t.unsqueeze(-1) * self.basis_freq.view(1, 1, -1) \
              + self.phase.view(1, 1, -1)
        return torch.cos(out)


class TKGTransformer(nn.Module):
    def __init__(self, n_concepts: int, n_edge_types: int, n_static: int,
                 n_classes: int = 6, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.concept_emb = nn.Embedding(n_concepts, d_model)
        self.edge_emb = nn.Embedding(n_edge_types, d_model)
        self.time_enc = TimeEncoder(d_model)
        self.value_proj = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(),
        )
        self.input_proj = nn.Linear(d_model * 4, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )

        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, concept_idx, edge_type_idx, t, v_norm, v_present,
                static, mask, return_attention: bool = False):
        ce = self.concept_emb(concept_idx)
        ee = self.edge_emb(edge_type_idx)
        te = self.time_enc(t)
        ve = self.value_proj(torch.stack([v_norm, v_present.float()], dim=-1))
        x = self.input_proj(torch.cat([ce, ee, te, ve], dim=-1))
        x = self.input_norm(x)

        kpm = ~mask
        x = self.encoder(x, src_key_padding_mask=kpm)

        q = self.pool_query.expand(x.size(0), -1, -1)
        pooled, attn_weights = self.pool_attn(
            q, x, x, key_padding_mask=kpm,
            need_weights=True, average_attn_weights=True,
        )
        pooled = pooled.squeeze(1)

        s = self.static_proj(static)
        out = self.head(torch.cat([pooled, s], dim=-1))
        if return_attention:
            return out, attn_weights.squeeze(1)
        return out


class PatientEventsDataset(Dataset):
    def __init__(self, subject_ids, events_by_sid, static_arr_by_sid,
                 label_by_sid, max_len: int = MAX_SEQ_LEN):
        self.sids = list(subject_ids)
        self.events = events_by_sid
        self.static = static_arr_by_sid
        self.labels = label_by_sid
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sids)

    def __getitem__(self, i: int):
        sid = self.sids[i]
        ev = self.events[sid]
        if len(ev) == 5:
            c, e, t, vn, vp = ev
        else:
            c, e, t = ev
            vn = np.zeros_like(t, dtype=np.float32)
            vp = np.zeros_like(t, dtype=np.float32)
        L = len(c)
        if L > self.max_len:
            c = c[-self.max_len:]
            e = e[-self.max_len:]
            t = t[-self.max_len:]
            vn = vn[-self.max_len:]
            vp = vp[-self.max_len:]
        return {
            "concept_idx": torch.from_numpy(c).long(),
            "edge_type_idx": torch.from_numpy(e).long(),
            "t": torch.from_numpy(t).float(),
            "v_norm": torch.from_numpy(np.asarray(vn, dtype=np.float32)),
            "v_present": torch.from_numpy(np.asarray(vp, dtype=np.float32)),
            "static": torch.from_numpy(self.static[sid]).float(),
            "label": int(self.labels[sid]),
            "len": min(L, self.max_len),
            "sid": int(sid),
        }


def collate(batch):
    Lmax = max(b["len"] for b in batch)
    B = len(batch)
    concept = torch.zeros(B, Lmax, dtype=torch.long)
    edge_t = torch.zeros(B, Lmax, dtype=torch.long)
    t = torch.zeros(B, Lmax, dtype=torch.float)
    v_norm = torch.zeros(B, Lmax, dtype=torch.float)
    v_present = torch.zeros(B, Lmax, dtype=torch.float)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    static = torch.stack([b["static"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    sids = torch.tensor([b["sid"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["len"]
        concept[i, :L]   = b["concept_idx"]
        edge_t[i, :L]    = b["edge_type_idx"]
        t[i, :L]         = b["t"]
        v_norm[i, :L]    = b["v_norm"]
        v_present[i, :L] = b["v_present"]
        mask[i, :L]      = True
    return {
        "concept_idx": concept, "edge_type_idx": edge_t,
        "t": t, "v_norm": v_norm, "v_present": v_present,
        "static": static, "mask": mask,
        "label": labels, "sid": sids,
    }


def _per_endpoint(y_idx: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for ep, j in EP_TO_IDX.items():
        y_bin = (y_idx == j).astype(int)
        if y_bin.sum() in (0, len(y_bin)):
            rows.append({"endpoint": ep, "auroc": np.nan,
                         "auprc": np.nan, "support": int(y_bin.sum())})
            continue
        auroc = roc_auc_score(y_bin, proba[:, j])
        auprc = average_precision_score(y_bin, proba[:, j])
        rows.append({"endpoint": ep, "auroc": auroc, "auprc": auprc,
                     "support": int(y_bin.sum())})
    return pd.DataFrame(rows)


def _overall(y_idx: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_idx, pred),
        "macro_f1": f1_score(y_idx, pred, average="macro"),
        "weighted_f1": f1_score(y_idx, pred, average="weighted"),
        "log_loss": log_loss(y_idx, proba,
                              labels=list(range(len(ENDPOINT_ORDER)))),
    }


def _set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _prepare_data():
    print("Loading modeling artifacts...")
    labels = pd.read_csv(os.path.join(MODELING_DIR, "labels.csv"))
    static = pd.read_csv(os.path.join(MODELING_DIR, "static_features.csv"))
    events = read_events_table()
    edge_types = pd.read_csv(os.path.join(MODELING_DIR, "edge_types.csv"))

    # Train-observed concepts only; OOV (test-only) routed to UNK at idx 0
    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    train_concepts = sorted(
        events.loc[events["subject_id"].isin(train_ids),
                    "concept_node_idx"].unique().tolist()
    )
    concept_remap = {c: i + 1 for i, c in enumerate(train_concepts)}
    n_concepts = len(train_concepts) + 1
    n_edge_types = len(edge_types)
    all_concepts = events["concept_node_idx"].nunique()
    print(f"  patients={len(labels):,}, events={len(events):,}")
    print(f"  concepts (train-only + UNK): {n_concepts:,} "
          f"(restricted from {all_concepts:,}); edge_types={n_edge_types}")

    events["c_emb_idx"] = events["concept_node_idx"].map(concept_remap).fillna(0).astype(int)
    n_oov = int((events["c_emb_idx"] == 0).sum())
    print(f"  OOV events mapped to UNK: {n_oov:,} "
          f"({n_oov / max(len(events), 1) * 100:.2f}% of events)")
    events = events.sort_values(["subject_id", "relative_days"]).reset_index(drop=True)

    if "value_norm" not in events.columns:
        events["value_norm"] = 0.0
    if "value_present" not in events.columns:
        events["value_present"] = 0.0
    events["value_norm"] = events["value_norm"].astype(np.float32)
    events["value_present"] = events["value_present"].astype(np.float32)

    events_by_sid: dict[int, tuple] = {}
    grouped = events.groupby("subject_id")
    for sid, g in grouped:
        events_by_sid[sid] = (
            g["c_emb_idx"].to_numpy(dtype=np.int64),
            g["edge_type_idx"].to_numpy(dtype=np.int64),
            g["relative_days"].to_numpy(dtype=np.float32),
            g["value_norm"].to_numpy(dtype=np.float32),
            g["value_present"].to_numpy(dtype=np.float32),
        )

    for sid in labels["subject_id"]:
        if sid not in events_by_sid:
            # A single UNK/no-value placeholder event, not a truly empty
            # sequence: collate() marks a fully-empty row's attention mask as
            # all-False, and a fully-masked row into nn.TransformerEncoder /
            # MultiheadAttention produces NaN (softmax over all -inf scores),
            # which would then poison the whole batch's loss via .mean().
            # concept_idx=0 is UNK, value_present=0 marks "no value" -- this
            # patient carries no real information either way.
            events_by_sid[sid] = (
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            )

    # Static feature normalization uses training statistics only
    static_cols = ["age_at_index", "cci_score", "num_cardiometa_conditions",
                   "had_icu_stay", "female"]
    train_ids = set(labels.loc[labels["split"] == "train", "subject_id"])
    static_train = static[static["subject_id"].isin(train_ids)][static_cols]
    mu = static_train.mean()
    sd = static_train.std().replace(0, 1.0)
    static_norm = (static[static_cols] - mu) / sd
    static_arr_by_sid = dict(zip(
        static["subject_id"].to_numpy(),
        static_norm.to_numpy(dtype=np.float32),
    ))

    # Append per-patient discharge-note BioBERT embedding to static block.
    # Controlled EXPLICITLY by env var TKG_USE_NOTES (not silently inferred from
    # file presence), so the strict (structured-only) and multimodal variants
    # are reproducible and unambiguous:
    #   TKG_USE_NOTES=0  -> strict, structured-only (no note block at all)
    #   TKG_USE_NOTES=1  -> multimodal (default; requires the note artifacts)
    use_notes = os.environ.get("TKG_USE_NOTES", "1").lower() not in ("0", "false", "no")
    notes_emb_path = os.path.join(OUTPUT_DIR, "notes", "patient_note_emb.npy")
    notes_csv_path = os.path.join(OUTPUT_DIR, "notes", "patient_note_emb.csv")
    n_static_total = len(static_cols)
    if not use_notes:
        print(f"  TKG_USE_NOTES=0 -> strict structured-only; n_static = {n_static_total}")
    elif os.path.exists(notes_emb_path) and os.path.exists(notes_csv_path):
        notes_emb = np.load(notes_emb_path).astype(np.float32)
        notes_meta = pd.read_csv(notes_csv_path)
        sid_to_note_row = {int(sid): i for i, sid
                            in enumerate(notes_meta["subject_id"])}
        has_notes_arr = notes_meta["has_notes"].to_numpy(dtype=np.float32)
        train_rows = [sid_to_note_row[int(s)] for s in train_ids
                       if int(s) in sid_to_note_row]
        if train_rows:
            train_emb = notes_emb[train_rows]
            emb_mu = train_emb.mean(axis=0)
            emb_sd = train_emb.std(axis=0)
            emb_sd[emb_sd == 0] = 1.0
            notes_emb_z = ((notes_emb - emb_mu) / emb_sd).astype(np.float32)
        else:
            notes_emb_z = notes_emb
        note_dim = notes_emb_z.shape[1]
        for sid, vec in list(static_arr_by_sid.items()):
            row = sid_to_note_row.get(int(sid))
            if row is not None:
                note_vec = notes_emb_z[row]
                hn = float(has_notes_arr[row])
            else:
                note_vec = np.zeros(note_dim, dtype=np.float32)
                hn = 0.0
            static_arr_by_sid[sid] = np.concatenate(
                [vec, np.array([hn], dtype=np.float32), note_vec]
            ).astype(np.float32)
        n_static_total += 1 + note_dim
        n_with = int(has_notes_arr.sum())
        print(f"  multimodal: appended has_notes + {note_dim}-d BioBERT to static; "
              f"{n_with:,} patients with notes "
              f"({n_with / max(len(notes_meta), 1) * 100:.1f}%); "
              f"n_static = {n_static_total}")
    else:
        print(f"  notes not found at {notes_emb_path}; running structured-only")

    label_by_sid = dict(zip(labels["subject_id"],
                              labels["endpoint_type"].map(EP_TO_IDX)))

    splits = {
        s: labels.loc[labels["split"] == s, "subject_id"].tolist()
        for s in ("train", "val", "test")
    }
    return (events_by_sid, static_arr_by_sid, label_by_sid,
            splits, n_concepts, n_edge_types, n_static_total, labels)


def _evaluate(model, loader, device, n_classes=6):
    model.eval()
    all_logits, all_y, all_sids = [], [], []
    with torch.no_grad():
        for batch in loader:
            for k in ("concept_idx", "edge_type_idx", "t",
                     "v_norm", "v_present", "static", "mask"):
                batch[k] = batch[k].to(device)
            logits = model(batch["concept_idx"], batch["edge_type_idx"],
                            batch["t"], batch["v_norm"], batch["v_present"],
                            batch["static"], batch["mask"])
            all_logits.append(logits.cpu())
            all_y.append(batch["label"])
            all_sids.append(batch["sid"])
    logits = torch.cat(all_logits)
    proba = F.softmax(logits, dim=-1).numpy()
    y = torch.cat(all_y).numpy()
    sids = torch.cat(all_sids).numpy()
    return proba, y, sids


def _plot_roc_compare(yte, tgn_proba, baseline_dir, out_path):
    """6-panel ROC: TGN vs LogReg + XGBoost per endpoint."""
    preds_b = pd.read_csv(os.path.join(baseline_dir, "predictions_test.csv"))
    preds_b = preds_b.set_index("subject_id")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for k, ep in enumerate(ENDPOINT_ORDER):
        ax = axes[k]
        j = EP_TO_IDX[ep]
        y_bin = (yte == j).astype(int)
        if y_bin.sum() == 0:
            ax.set_title(f"{ep} (no positives)")
            continue
        # Note: tgn_proba is aligned to test sids; ensure same order
        for name, proba in [("LogReg", preds_b[f"logreg_p_{ep}"].to_numpy()),
                             ("XGBoost", preds_b[f"xgb_p_{ep}"].to_numpy()),
                             ("TGN", tgn_proba[:, j])]:
            fpr, tpr, _ = roc_curve(y_bin, proba)
            try:
                auc = roc_auc_score(y_bin, proba)
            except ValueError:
                auc = float("nan")
            lw = 2.0 if name == "TGN" else 1.0
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=lw)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5)
        ax.set_title(f"{ep} (n={int(y_bin.sum())})", fontweight="bold")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Test-set ROC: baselines vs TGN",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_and_eval() -> None:
    _set_seed()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    (events_by_sid, static_arr_by_sid, label_by_sid,
     splits, n_concepts, n_edge_types, n_static,
     labels_df) = _prepare_data()

    # Datasets and loaders
    def _make_loader(sids, shuffle):
        ds = PatientEventsDataset(sids, events_by_sid, static_arr_by_sid,
                                    label_by_sid, max_len=MAX_SEQ_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                            collate_fn=collate, num_workers=0)

    train_loader = _make_loader(splits["train"], shuffle=True)
    val_loader = _make_loader(splits["val"], shuffle=False)
    test_loader = _make_loader(splits["test"], shuffle=False)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  device: {device}")

    # Model
    model = TKGTransformer(
        n_concepts=n_concepts, n_edge_types=n_edge_types,
        n_static=n_static, n_classes=len(ENDPOINT_ORDER),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    # Class-balanced cross-entropy
    train_labels = np.array([label_by_sid[s] for s in splits["train"]])
    counts = np.bincount(train_labels, minlength=len(ENDPOINT_ORDER)).astype(float)
    inv = (1.0 / counts) * counts.sum() / len(ENDPOINT_ORDER)
    cw = torch.tensor(inv, dtype=torch.float32, device=device)
    print(f"  class weights: {dict(zip(ENDPOINT_ORDER, inv.round(3)))}")
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    optim = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_val_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    history = []
    best_state = None

    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE})...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            for k in ("concept_idx", "edge_type_idx", "t",
                     "v_norm", "v_present", "static", "mask"):
                batch[k] = batch[k].to(device)
            batch["label"] = batch["label"].to(device)
            optim.zero_grad()
            logits = model(batch["concept_idx"], batch["edge_type_idx"],
                            batch["t"], batch["v_norm"], batch["v_present"],
                            batch["static"], batch["mask"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        train_loss /= max(n_batches, 1)

        # Val
        val_proba, val_y, _ = _evaluate(model, val_loader, device)
        val_macro = f1_score(val_y, val_proba.argmax(axis=1), average="macro")
        val_acc = accuracy_score(val_y, val_proba.argmax(axis=1))
        val_per_ep = _per_endpoint(val_y, val_proba)
        val_mean_auc = val_per_ep[val_per_ep["endpoint"] != "censored"]["auroc"].mean()
        dt = time.time() - t0
        print(f"  ep {epoch:02d}  loss={train_loss:.4f}  "
              f"val_macroF1={val_macro:.4f}  val_acc={val_acc:.4f}  "
              f"val_AUROC(mean)={val_mean_auc:.4f}  ({dt:.1f}s)")
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_macro_f1": val_macro, "val_acc": val_acc,
            "val_mean_auroc": val_mean_auc, "time_s": dt,
        })
        if val_macro > best_val_f1:
            best_val_f1 = val_macro
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stop at epoch {epoch} "
                      f"(best epoch {best_epoch}, val macroF1={best_val_f1:.4f})")
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    print("\nEvaluating on test set...")
    test_proba, test_y, test_sids = _evaluate(model, test_loader, device)
    per_ep = _per_endpoint(test_y, test_proba)
    overall = _overall(test_y, test_proba)
    print("\n=== TGN TEST METRICS ===")
    print("  endpoint    AUROC    AUPRC    n_pos")
    for r in per_ep.itertuples(index=False):
        print(f"  {r.endpoint:<10s} {r.auroc:8.3f} {r.auprc:8.3f} {r.support:8d}")
    print(f"  overall: acc={overall['accuracy']:.3f}  "
          f"macroF1={overall['macro_f1']:.3f}  "
          f"weightedF1={overall['weighted_f1']:.3f}  "
          f"logloss={overall['log_loss']:.3f}")

    # Save artifacts
    print("\nSaving artifacts...")
    rows = []
    for split_name, (y_eval, proba) in [
        ("val", (val_y, val_proba)),
        ("test", (test_y, test_proba)),
    ]:
        ov = _overall(y_eval, proba)
        per = _per_endpoint(y_eval, proba)
        for r in per.itertuples(index=False):
            rows.append({"model": "tgn", "split": split_name,
                         "endpoint": r.endpoint,
                         "auroc": r.auroc, "auprc": r.auprc,
                         "support": r.support, **ov})
    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    preds = pd.DataFrame({
        "subject_id": test_sids,
        "endpoint_true": pd.Series(test_y).map({i: ep for ep, i in EP_TO_IDX.items()}).values,
    })
    for j, ep in enumerate(ENDPOINT_ORDER):
        preds[f"tgn_p_{ep}"] = test_proba[:, j]
    preds_path = os.path.join(MODEL_DIR, "predictions_test.csv")
    preds.to_csv(preds_path, index=False)

    hist_path = os.path.join(MODEL_DIR, "history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)

    ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
    torch.save({"state_dict": best_state, "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "config": {"d_model": D_MODEL, "n_heads": N_HEADS,
                            "n_layers": N_LAYERS, "max_seq_len": MAX_SEQ_LEN}},
                ckpt_path)

    # Comparison figure
    roc_path = os.path.join(FIGURES_DIR, "fig6_tgn_vs_baseline_roc.png")
    # Reorder TGN predictions to match baseline predictions' subject order
    baseline_pred = pd.read_csv(
        os.path.join(OUTPUT_DIR, "baseline", "predictions_test.csv"))
    order = baseline_pred["subject_id"].to_numpy()
    sid_to_idx = {int(s): i for i, s in enumerate(test_sids)}
    tgn_proba_ordered = test_proba[[sid_to_idx[int(s)] for s in order]]
    yte_ordered = test_y[[sid_to_idx[int(s)] for s in order]]
    _plot_roc_compare(yte_ordered, tgn_proba_ordered,
                       os.path.join(OUTPUT_DIR, "baseline"), roc_path)

    print(f"  {metrics_path}")
    print(f"  {preds_path}")
    print(f"  {hist_path}")
    print(f"  {ckpt_path}")
    print(f"  {roc_path}")


if __name__ == "__main__":
    train_and_eval()
