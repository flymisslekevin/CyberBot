"""Quick command‑line helper to grep your mapping.json file.

Usage:
    python check_mapping.py "routine dental"            # regex, case‑insensitive
    python check_mapping.py "Part B deductible" 10       # regex + max hits

Place this file in the same directory as rag_local.py so that the
relative path to vectors/mapping.json remains valid.
"""

import json, re, sys, textwrap, pathlib

MAPPING_PATH = "vectors/mapping.json"


def search_mapping(pattern: str, max_hits: int = 5, flags=re.I):
    """Return a list of (idx, metadata) hits whose .text matches pattern."""
    regex = re.compile(pattern, flags)

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    hits = [(idx, meta) for idx, meta in enumerate(mapping) if regex.search(meta["text"])]

    print(f"Found {len(hits)} hits for /{pattern}/\n")
    for idx, meta in hits[:max_hits]:
        src   = meta.get("source", "?")
        page  = meta.get("page", "?")
        text  = meta["text"].replace("\n", " ")
        snippet = textwrap.shorten(text, width=400, placeholder="…")
        print(f"ID {idx}  {src} p.{page}")
        print(textwrap.fill(snippet, width=100))
        print("-" * 80)

    return hits


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_mapping.py <regex pattern> [max_hits]")
        sys.exit(1)

    pattern = sys.argv[1]
    max_hits = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    search_mapping(pattern, max_hits)
