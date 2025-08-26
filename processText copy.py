#!/usr/bin/env python3
"""
process_local_pdfs.py

Walks through three local folders under `pdf_storage`:
  - industryReports
  - regulations
  - securityFrameworks

Extracts text from every PDF and writes a .txt into a
single `pdf_storage/processed/` folder, mirroring the
original subfolder structure.

Prereq: pip install pdfplumber
"""

import traceback
from pathlib import Path
import pdfplumber

# ── 1. Configure your local storage root ───────────────────────────────────────
BASE_DIR      = Path("pdf_storage")
SUBFOLDERS    = ["industryReports", "regulations", "securityFrameworks"]
PROCESSED_ROOT = BASE_DIR / "processed"


# ── 2. PDF → text helper ───────────────────────────────────────────────────────
def extract_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


# ── 3. Main loop ───────────────────────────────────────────────────────────────
def process_pdfs():
    # ensure the top-level processed folder exists
    PROCESSED_ROOT.mkdir(exist_ok=True)

    for sub in SUBFOLDERS:
        src_dir = BASE_DIR / sub

        for pdf_file in src_dir.rglob("*.pdf"):
            # rel = industryReports/foo.pdf  or regulations/bar.pdf, etc.
            rel      = pdf_file.relative_to(BASE_DIR).with_suffix(".txt")
            out_path = PROCESSED_ROOT / rel

            # make sure parent folders exist under processed/
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                print(f"✓ Skipping (exists): {out_path}")
                continue

            try:
                text = extract_text(pdf_file)
                out_path.write_text(text, encoding="utf-8")
                print(f"✅ {pdf_file.name} → {out_path}")
            except Exception:
                print(f"⚠️ Error processing {pdf_file}")
                traceback.print_exc()


if __name__ == "__main__":
    process_pdfs()
