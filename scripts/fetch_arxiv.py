"""
Fetch open-access papers from arXiv to extend the library.

Two modes:

    # discover candidates without downloading anything
    python scripts/fetch_arxiv.py --search "cat:cs.CL AND abs:\\"hate speech\\"" --max 8 --dry-run

    # download a pinned set, which is what the eval corpus should use
    python scripts/fetch_arxiv.py --ids 2408.08921 1810.04805

Pinned IDs matter: a golden question set is only reproducible if everyone ends
up with the same PDFs, and a search re-run months later returns different ones.
`--search` is for finding candidates; `--ids` is for building the corpus.

Downloads are written to papers/ and a manifest recording id, title, categories
and published date goes to papers/arxiv_manifest.json, so the corpus can be
rebuilt from scratch on another machine.

arXiv asks for no more than one request every three seconds; this obeys that.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# arXiv's stated rate limit for the public API.
POLITE_DELAY = 3.0
UA = "research-rag/0.2 (personal research corpus builder)"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _text(entry, path: str) -> str:
    el = entry.find(path, NS)
    return " ".join(el.text.split()) if el is not None and el.text else ""


def query(search: str | None, ids: list[str] | None, max_results: int) -> list[dict]:
    params = {"max_results": str(max_results), "start": "0"}
    if ids:
        params["id_list"] = ",".join(ids)
    else:
        params["search_query"] = search or ""
        params["sortBy"] = "relevance"

    raw = _get(f"{API}?{urllib.parse.urlencode(params)}")
    root = ET.fromstring(raw)

    out = []
    for e in root.findall("a:entry", NS):
        full_id = _text(e, "a:id").rsplit("/", 1)[-1]        # e.g. 2408.08921v2
        out.append({
            "id": full_id,
            "base_id": re.sub(r"v\d+$", "", full_id),
            "title": _text(e, "a:title"),
            "published": _text(e, "a:published")[:10],
            "primary": (e.find("arxiv:primary_category", NS).get("term")
                        if e.find("arxiv:primary_category", NS) is not None else ""),
            "categories": [c.get("term") for c in e.findall("a:category", NS)],
            "summary": _text(e, "a:summary"),
            "pdf": next((l.get("href") for l in e.findall("a:link", NS)
                         if l.get("type") == "application/pdf"), None),
        })
    return out


def download(meta: dict, out_dir: Path) -> Path | None:
    url = meta.get("pdf") or f"https://arxiv.org/pdf/{meta['id']}"
    # Name by arXiv id: stable, unambiguous, and safe on every filesystem.
    dest = out_dir / f"arxiv_{meta['base_id'].replace('/', '_')}.pdf"
    if dest.exists():
        print(f"  = {dest.name} (already downloaded)")
        return dest
    try:
        data = _get(url, timeout=120)
    except Exception as exc:
        print(f"  ! {meta['base_id']}: {exc}", file=sys.stderr)
        return None
    if not data.startswith(b"%PDF"):
        print(f"  ! {meta['base_id']}: not a PDF ({len(data)} bytes)", file=sys.stderr)
        return None
    dest.write_bytes(data)
    print(f"  + {dest.name}  ({len(data) // 1024} KB)  {meta['title'][:52]}")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--search", help='arXiv query, e.g. \'cat:math.PR AND abs:"martingale"\'')
    g.add_argument("--ids", nargs="+", help="Explicit arXiv ids to download")
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "papers"))
    ap.add_argument("--dry-run", action="store_true", help="List matches, download nothing")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = query(args.search, args.ids, args.max)
    if not metas:
        print("No results.")
        return

    print(f"{len(metas)} result(s):\n")
    for m in metas:
        print(f"  {m['base_id']:14s} {m['primary']:10s} {m['published']}  {m['title'][:66]}")

    if args.dry_run:
        print("\nDry run - nothing downloaded.")
        return

    print("\nDownloading:")
    manifest_path = out_dir / "arxiv_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for i, m in enumerate(metas):
        dest = download(m, out_dir)
        if dest:
            manifest[dest.name] = {
                k: m[k] for k in ("id", "base_id", "title", "published", "primary", "categories")
            }
        if i < len(metas) - 1:
            time.sleep(POLITE_DELAY)

    manifest_path.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nManifest: {manifest_path} ({len(manifest)} paper(s))")
    print("Ingest with:  python -m research_rag.cli ingest papers/")


if __name__ == "__main__":
    main()
