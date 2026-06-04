"""Section-aware markdown chunker for clinical protocol documents.

Splits markdown by H2 headings, then packs each section into chunks of
roughly `target_tokens` with `overlap_tokens` overlap. Attaches metadata
so the downstream RAG pipeline can filter by `doc_type` (e.g.
'protocol_synopsis' vs 'crf_demographics') or `section` (e.g.
'Inclusion Criteria') without re-running the embedding query.

This is the iteration-2 replacement for the iteration-1 TF-IDF mock —
chunks live in a Delta table that the Vector Search Delta Sync index
embeds and serves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str       # e.g. "protocol_synopsis::study_design::0"
    doc_type: str       # filename stem, e.g. "protocol_synopsis"
    section: str        # H2 heading text, e.g. "Study Design"
    chunk_index: int    # position within section (0-based)
    chunk_text: str
    token_count: int    # approximate; words * 1.3
    source_path: str    # relative path from project root


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _split_by_section(md_text: str) -> list[tuple[str, str]]:
    """Split markdown by H2 headings. Returns [(section_title, body)]."""
    sections: list[tuple[str, str]] = []
    current_title = "Preamble"
    current_body: list[str] = []
    for line in md_text.split("\n"):
        m = re.match(r"^##\s+(.+)$", line)
        if m:
            if any(s.strip() for s in current_body):
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = m.group(1).strip()
            current_body = []
        else:
            current_body.append(line)
    if any(s.strip() for s in current_body):
        sections.append((current_title, "\n".join(current_body).strip()))
    return [(t, b) for t, b in sections if b]


def _pack_section(section_text: str, target_tokens: int = 800,
                  overlap_tokens: int = 150) -> list[str]:
    """If a section fits, return it whole; else pack with overlap."""
    if _approx_tokens(section_text) <= target_tokens:
        return [section_text]
    words = section_text.split()
    target_words = int(target_tokens / 1.3)
    overlap_words = int(overlap_tokens / 1.3)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def chunk_doc(md_path: Path, project_root: Path,
              target_tokens: int = 800, overlap_tokens: int = 150) -> list[Chunk]:
    md_text = md_path.read_text()
    doc_type = md_path.stem
    rel_path = str(md_path.relative_to(project_root))
    chunks: list[Chunk] = []
    for section_title, section_text in _split_by_section(md_text):
        slug = re.sub(r"[^a-z0-9]+", "_", section_title.lower()).strip("_")[:40]
        for i, body in enumerate(
            _pack_section(section_text, target_tokens, overlap_tokens)
        ):
            chunks.append(Chunk(
                chunk_id=f"{doc_type}::{slug}::{i}",
                doc_type=doc_type,
                section=section_title,
                chunk_index=i,
                chunk_text=body,
                token_count=_approx_tokens(body),
                source_path=rel_path,
            ))
    return chunks


def chunk_all(docs_dir: Path, project_root: Path | None = None) -> list[Chunk]:
    project_root = project_root or docs_dir.parent.parent
    chunks: list[Chunk] = []
    for md_path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_doc(md_path, project_root))
    return chunks
