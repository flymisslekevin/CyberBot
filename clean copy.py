#!/usr/bin/env python3
"""
clean_local.py

Recursively cleans text files under `pdf_storage/processed/`
and writes cleaned versions to `pdf_storage/clean/`,
preserving the original folder structure.

Prereq: pip install pdfplumber
"""

import re
import html
import traceback
from pathlib import Path

# ── 1. Configuration ─────────────────────────────────────────────────────────
BASE_DIR    = Path("pdf_storage")
SRC_DIR     = BASE_DIR / "processed"
DST_DIR     = BASE_DIR / "clean"

# ── 2. Cleaning function ─────────────────────────────────────────────────────
def basic_clean(text: str) -> str:
    # Unescape HTML entities
    text = html.unescape(text)
    # Collapse any whitespace sequence into a single space
    text = re.sub(r"\s+", " ", text)
    # Remove common footer patterns like "Page X of Y"
    text = re.sub(r"Page \d+ of \d+", "", text)
    return text.strip()

# ── 3. Main processing ─────────────────────────────────────────────────────────
def clean_all():
    # Ensure the cleaned folder root exists
    DST_DIR.mkdir(exist_ok=True)

    for txt_file in SRC_DIR.rglob("*.txt"):
        # Compute the relative path under processed/
        rel_path = txt_file.relative_to(SRC_DIR)
        out_file = DST_DIR / rel_path
        # Create parent directories under clean/
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already cleaned
        if out_file.exists():
            print(f"✓ Skipping (exists): {rel_path}")
            continue

        try:
            raw_text = txt_file.read_text(encoding="utf-8")
            cleaned  = basic_clean(raw_text)
            out_file.write_text(cleaned, encoding="utf-8")
            print(f"✅ Cleaned: {rel_path} → {out_file}")
        except Exception as e:
            print(f"⚠️ Error processing {rel_path}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    clean_all()
