"""Embed cohort discharge notes with Bio_ClinicalBERT.

Uses the [CLS] token output (768-dim) as a per-note embedding.
Embeddings get used in the TGN as event features for `hasNote` facts.

Output:
  tkg_output/notes/note_embeddings.npy   -- (N, 768) float32
  tkg_output/notes/note_ids.csv          -- aligned note_id ordering
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from src.config import OUTPUT_DIR

NOTES_DIR = os.path.join(OUTPUT_DIR, "notes")
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
BATCH_SIZE = 32
EMB_DIM = 768


def embed_notes() -> None:
    os.makedirs(NOTES_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME} (first run downloads ~440 MB)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    print(f"  device: {device}")

    print("\nLoading cohort notes...")
    notes = pd.read_csv(
        os.path.join(NOTES_DIR, "discharge_cohort.csv.gz"),
        usecols=["note_id", "text"],
    )
    # Sort by note_id so embedding order is deterministic & joinable
    notes = notes.sort_values("note_id").reset_index(drop=True)
    note_ids = notes["note_id"].tolist()
    texts = notes["text"].fillna("").astype(str).tolist()
    print(f"  notes to embed: {len(texts):,}")

    out_path = os.path.join(NOTES_DIR, "note_embeddings.npy")
    ids_path = os.path.join(NOTES_DIR, "note_ids.csv")

    # Resume support: if a partial file exists, skip already-embedded rows
    start_idx = 0
    if os.path.exists(out_path) and os.path.exists(ids_path):
        existing_ids = pd.read_csv(ids_path)["note_id"].tolist()
        if existing_ids == note_ids[:len(existing_ids)]:
            start_idx = len(existing_ids)
            embeddings = np.load(out_path)
            assert embeddings.shape == (len(note_ids), EMB_DIM), \
                "Existing embeddings shape mismatch; delete to restart."
            print(f"  resuming from index {start_idx:,}")
        else:
            print("  existing IDs don't match current order; restarting")
            embeddings = np.zeros((len(note_ids), EMB_DIM), dtype=np.float32)
    else:
        embeddings = np.zeros((len(note_ids), EMB_DIM), dtype=np.float32)
    # First-time: persist ids so resume works on the next run
    pd.DataFrame({"note_id": note_ids}).to_csv(ids_path, index=False)

    print(f"\nEmbedding (batch={BATCH_SIZE}, max_tokens={MAX_LEN})...")
    t0 = time.time()
    save_every = 50  # batches between checkpoints
    with torch.no_grad():
        for i in range(start_idx, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LEN,
            ).to(device)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings[i:i + len(batch_texts)] = cls
            batch_idx = i // BATCH_SIZE
            if batch_idx % 20 == 0:
                done = i + len(batch_texts)
                dt = time.time() - t0
                rate = max((done - start_idx) / max(dt, 0.001), 1e-6)
                eta = (len(texts) - done) / rate / 60.0
                print(f"  {done:>7,}/{len(texts):,}  "
                      f"({rate:.1f} notes/s, ETA {eta:.1f} min)",
                      flush=True)
            if batch_idx % save_every == 0:
                np.save(out_path, embeddings)

    np.save(out_path, embeddings)
    dt = (time.time() - t0) / 60.0
    print(f"\nDone in {dt:.1f} min")
    print(f"\nSaved:\n  {out_path}  ({embeddings.shape})\n  {ids_path}")


if __name__ == "__main__":
    embed_notes()
