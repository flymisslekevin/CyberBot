#!/usr/bin/env python3
"""
build_local_index.py

Scans cleaned text files under `pdf_storage/clean/`, chunks text,
embeds via Ollama, and builds a brand‐new FAISS index + mapping.json
in your local `vectors/` directory.

Prereqs:
  pip install tqdm numpy faiss-cpu ollama transformers

Usage:
  python build_local_index.py
"""

import json
import pathlib
import numpy as np
import faiss
import ollama
from tqdm import tqdm
from transformers import GPT2TokenizerFast

# ── 1. Configuration ──────────────────────────────────────────────────────────
BASE_DIR      = pathlib.Path("pdf_storage") / "clean"
VEC_DIR       = pathlib.Path("vectors")
INDEX_PATH    = VEC_DIR / "faiss.index"
MAPPING_PATH  = VEC_DIR / "mapping.json"
EMB_MODEL     = "nomic-embed-text"
CHUNK_TOK     = 500
OVERLAP       = 50

# Ensure vectors directory exists
VEC_DIR.mkdir(parents=True, exist_ok=True)

# ── 2. Tokenizer setup ────────────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# ── 3. Read, chunk, embed ─────────────────────────────────────────────────────
all_vecs = []
mapping  = []

print("🔹 Scanning cleaned .txt files and embedding…")
for txt_file in tqdm(list(BASE_DIR.rglob("*.txt")), desc="Files"):
    # derive an ID and original PDF source path
    rel = txt_file.relative_to(BASE_DIR)               # e.g. industryReports/foo.txt
    stem = rel.with_suffix("").as_posix()              # e.g. "industryReports/foo"
    source_pdf = rel.with_suffix(".pdf").as_posix()    # e.g. "industryReports/foo.pdf"

    text = txt_file.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    i = 0
    idx = 0

    while i < len(token_ids):
        chunk_ids = token_ids[i : i + CHUNK_TOK]
        passage   = tokenizer.decode(chunk_ids)

        # embed
        vec = ollama.embeddings(model=EMB_MODEL, prompt=passage)["embedding"]
        all_vecs.append(vec)

        # record metadata
        mapping.append({
            "id": f"{stem}#{idx}",
            "text": passage,
            "source": source_pdf,
        })

        idx += 1
        i   += CHUNK_TOK - OVERLAP

# ── 4. Build FAISS index ───────────────────────────────────────────────────────
if not all_vecs:
    print("⚠️  No embeddings generated. Check `pdf_storage/clean/` folder.")
    exit(1)

arr = np.array(all_vecs, dtype="float32")
faiss.normalize_L2(arr)
dim   = arr.shape[1]
index = faiss.IndexFlatIP(dim)    # Inner product on normalized vectors = cosine sim
index.add(arr)

# ── 5. Persist index + mapping ────────────────────────────────────────────────
faiss.write_index(index, str(INDEX_PATH))
with open(MAPPING_PATH, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f"✅ Built fresh index with {index.ntotal} vectors")
print(f"   • faiss.index → {INDEX_PATH}")
print(f"   • mapping.json → {MAPPING_PATH}")
