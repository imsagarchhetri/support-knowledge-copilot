"""Document loaders + chunking.

Loaders for Markdown, HTML, plain text, and PDF. HTML/PDF parsers are imported
lazily so the core (md/txt) runs without those extras installed. Headings and
source metadata are preserved so retrieval can filter and cite precisely.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import get_settings
from .models import Chunk


# ---------------- loaders: file -> (clean_text, metadata) ----------------
def _front_matter(raw: str) -> dict:
    meta = {}
    for m in re.finditer(r"<!--\s*(\w+):\s*(.+?)\s*-->", raw):
        meta[m.group(1)] = m.group(2)
    return meta


def load_markdown(path: Path) -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8")
    return raw, _front_matter(raw)


def load_text(path: Path) -> tuple[str, dict]:
    return path.read_text(encoding="utf-8"), {}


def load_html(path: Path) -> tuple[str, dict]:
    from bs4 import BeautifulSoup  # lazy
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    # convert headings to markdown so the chunker can use them
    for level in range(1, 4):
        for tag in soup.find_all(f"h{level}"):
            tag.replace_with("\n" + "#" * level + " " + tag.get_text() + "\n")
    return soup.get_text(), {}


def load_pdf(path: Path) -> tuple[str, dict]:
    import fitz  # PyMuPDF, lazy
    parts = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, 1):
            parts.append(f"\n## Page {i}\n{page.get_text()}")
    return "\n".join(parts), {}


LOADERS = {".md": load_markdown, ".markdown": load_markdown, ".txt": load_text,
           ".html": load_html, ".htm": load_html, ".pdf": load_pdf}


# ---------------- chunking ----------------
def chunk_heading(text: str) -> list[dict]:
    parts = re.split(r"(?m)^(#{1,3}\s.+)$", text)
    out, heading = [], "Introduction"
    intro = parts[0].strip()
    if intro:
        out.append({"text": intro, "heading": heading})
    for i in range(1, len(parts), 2):
        heading = re.sub(r"^#+\s", "", parts[i]).strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            out.append({"text": f"{heading}\n{body}", "heading": heading})
    return out


def chunk_fixed(text: str, size: int, overlap: int) -> list[dict]:
    words = text.split()
    out, step = [], max(1, size - overlap)
    for i in range(0, len(words), step):
        body = " ".join(words[i:i + size])
        if body:
            out.append({"text": body, "heading": "window"})
    return out


def load_and_chunk(docs_dir: str | None = None,
                   strategy: str | None = None) -> list[Chunk]:
    s = get_settings()
    docs_dir = Path(docs_dir or s.docs_dir)
    strategy = strategy or s.chunk_strategy
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        loader = LOADERS.get(path.suffix.lower())
        if not loader:
            continue
        raw, meta = loader(path)
        pieces = (chunk_heading(raw) if strategy == "heading"
                  else chunk_fixed(raw, s.chunk_size, s.chunk_overlap))
        for j, p in enumerate(pieces):
            chunks.append(Chunk(
                chunk_id=f"{path.stem}::{strategy}::{j}",
                text=p["text"], source=path.name, heading=p["heading"],
                doc_type=meta.get("doc_type", "doc"),
                last_updated=meta.get("last_updated", "unknown"),
                access_level=meta.get("access_level", "internal"),
                strategy=strategy,
            ))
    return chunks
