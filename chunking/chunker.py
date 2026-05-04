"""
Resume Hierarchical Chunker
============================
Takes section-level JSON (from extract.py) and produces hierarchical chunks
with metadata, ready for embedding and storage in Qdrant.

Two levels:
  Level 1 ("section")   – full section text (e.g., all Projects)
  Level 2 ("sub_item")  – individual entries (e.g., one specific project)

Sections that get sub-chunked: Projects, Experience, and related variants.
All other sections remain as single section-level chunks.

Usage:
    python chunker.py sections.json -o chunks.json
    python chunker.py sections.json --resume-id saiteja_01

Pipeline:
    python extract.py resume.pdf -o sections.json
    python chunker.py sections.json -o chunks.json
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Sections that should be split into sub-items
# ---------------------------------------------------------------------------
HIERARCHICAL_SECTIONS = {
    "projects",
    "academic projects",
    "personal projects",
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "work history",
    "internships",
    "internship experience",
    "research",
    "research experience",
}

# Compiled patterns
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
BULLET_RE = re.compile(r"^\s*[•\-–—\*►]\s")


# ---------------------------------------------------------------------------
# Entry splitting logic
# ---------------------------------------------------------------------------

def _looks_like_entry_header(line: str) -> bool:
    """
    Decide whether a non-bullet line is an entry header vs. a PDF
    continuation (wrapped) line.

    Entry headers typically:
      • Start with an uppercase letter or digit
      • Contain a pipe '|'  (tech stack separator)
      • Contain a year like 2024
      • Are relatively short (titles, not mid-sentence text)
      • Don't start with common lowercase continuation words

    PDF continuations typically:
      • Start with a lowercase letter
      • Look like sentence fragments
      • Are long, flowing text
    """
    stripped = line.strip()
    if not stripped:
        return False

    # A standalone year line (e.g. "2025") is part of the CURRENT entry header,
    # not a new entry.  We handle it separately — it's NOT an entry boundary.
    if YEAR_RE.fullmatch(stripped):
        return False

    # Must start with uppercase letter or digit to be a header
    if not stripped[0].isupper() and not stripped[0].isdigit():
        return False

    # Strong signals that this IS a header
    has_pipe = "|" in stripped
    has_year = bool(YEAR_RE.search(stripped))

    if has_pipe:
        return True          # "Float Chat | Python, LangGraph"
    if has_year and len(stripped) < 120:
        return True          # "Software Engineer at Google   2023 - 2025"

    # Weaker signal: short, title-like line that doesn't look like a sentence
    # Sentence fragments usually end with common words or have many spaces
    if len(stripped) < 80 and not stripped.endswith((".","!","?",",")):
        # Check it's not a wrapped continuation of a sentence
        # Continuations rarely contain dates or common title patterns
        return True

    return False


def split_into_entries(content: str) -> list[str]:
    """
    Split a section's content into individual entries (projects / jobs).

    Heuristic
    ---------
    Resume entries universally follow this shape:

        Title / Company / Date      ← non-bullet "header" lines
        • first bullet description
        • second bullet description
        Next Title                  ← NEW ENTRY starts here

    A new entry boundary is detected when ALL of these are true:
      1. The line is non-bullet and non-blank
      2. At least one bullet line exists in the current entry (we're past the header)
      3. The line looks like a real entry header (starts uppercase, contains
         pipe or year, etc.) — NOT a PDF-wrapped continuation line

    This handles:
      • Single-line headers  ("Float Chat | Python, LangGraph     2025")
      • Multi-line headers   ("Software Engineer\\nGoogle, CA\\n2023-2025")
      • Standalone year line ("Float Chat | Python\\n2025\\n– Built ...")
      • PDF-wrapped bullets  ("– Long description that wraps onto\\nthe next line")
      • Various bullet styles (-, –, •, *, ►)
    """
    lines = content.split("\n")
    if not lines:
        return [content] if content.strip() else []

    entry_starts: list[int] = [0]  # first line always starts an entry

    for i in range(1, len(lines)):
        line = lines[i].strip()

        # Skip blank lines and bullet lines — they never start entries
        if not line or BULLET_RE.match(line):
            continue

        # Non-bullet, non-blank line found.
        # Check whether we've already seen bullets since the last entry start.
        preceding_has_bullets = any(
            BULLET_RE.match(lines[j].strip())
            for j in range(entry_starts[-1], i)
            if lines[j].strip()
        )

        if preceding_has_bullets and _looks_like_entry_header(line):
            entry_starts.append(i)

    # Build entry texts from the detected boundaries
    entries: list[str] = []
    for idx in range(len(entry_starts)):
        start = entry_starts[idx]
        end = entry_starts[idx + 1] if idx + 1 < len(entry_starts) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        if text:
            entries.append(text)

    return entries


def extract_entry_title(entry_text: str) -> str:
    """
    Pull a short, human-readable title from an entry's first non-bullet line.

    Examples:
      "Float Chat | Python, LangGraph, Gemini  2025"  →  "Float Chat"
      "Software Engineer at Google"                    →  "Software Engineer at Google"
      "B.Tech CSE at Bennett University  2023-2027"    →  "B.Tech CSE at Bennett University"
    """
    for line in entry_text.split("\n"):
        stripped = line.strip()
        if not stripped or BULLET_RE.match(stripped):
            continue

        # Take the portion before the first pipe '|'
        title = re.split(r"\|", stripped)[0].strip()

        # Remove trailing year and everything after it
        title = re.sub(r"\s*\b(19|20)\d{2}\b.*$", "", title).strip()

        # Remove trailing punctuation
        title = title.rstrip(" ,–—-:•|")

        if title:
            return title[:80]  # cap length for readability

    return "Untitled"


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------

def generate_chunk_id(resume_id: str, section: str, sub_section: str = "") -> str:
    """Deterministic ID so re-running produces the same IDs (idempotent upserts)."""
    raw = f"{resume_id}::{section}::{sub_section}".lower()
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def build_chunks(
    sections,
    resume_id="unknown",
    person_name=None,
):
    """
    Convert section-level data into hierarchical chunks.

    Parameters
    ----------
    sections : list[dict]
        Output of extract.py — each dict has "section" and "content" keys.
    resume_id : str
        Unique identifier for the resume (used in chunk IDs and metadata).
    person_name : str or None
        Candidate name.  Auto-detected from Personal Info if not given.

    Returns
    -------
    list[dict]
        Flat list of chunk dicts, each containing:
          chunk_id, resume_id, person_name, section, sub_section,
          level ("section" | "sub_item"), content, content_with_context
    """
    # --- Auto-detect person name from Personal Info section ---
    if person_name is None:
        for sec in sections:
            if sec["section"].lower() in (
                "personal info",
                "personal information",
                "contact",
                "contact information",
            ):
                first_line = sec["content"].split("\n")[0].strip()
                # Heuristic: name is the first line if it's not an email/URL
                if first_line and "@" not in first_line and not first_line.startswith("http"):
                    person_name = first_line
                    break
        if person_name is None:
            person_name = "Unknown"

    chunks: list[dict] = []

    for sec in sections:
        section_name = sec["section"]
        content = sec["content"]

        if not content.strip():
            continue

        # ── Level 1: Section-level chunk (always created) ──────────────
        chunks.append({
            "chunk_id": generate_chunk_id(resume_id, section_name),
            "resume_id": resume_id,
            "person_name": person_name.lower(),
            "section": section_name.lower(),
            "sub_section": "",
            "level": "section",
            "content": content,
            "content_with_context": (
                f"Resume: {person_name} | Section: {section_name}\n\n{content}"
            ),
        })

        # ── Level 2: Sub-item chunks (only for hierarchical sections) ─
        if section_name.lower() in HIERARCHICAL_SECTIONS:
            entries = split_into_entries(content)

            # Only create sub-items if there are 2+ entries.
            # A single entry would be identical to the section chunk.
            if len(entries) >= 2:
                for entry_text in entries:
                    title = extract_entry_title(entry_text)
                    chunks.append({
                        "chunk_id": generate_chunk_id(
                            resume_id, section_name, title
                        ),
                        "resume_id": resume_id,
                        "person_name": person_name.lower(),
                        "section": section_name.lower(),
                        "sub_section": title.lower(),
                        "level": "sub_item",
                        "content": entry_text,
                        "content_with_context": (
                            f"Resume: {person_name} | "
                            f"Section: {section_name} | "
                            f"Entry: {title}\n\n{entry_text}"
                        ),
                    })

    return chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert resume sections into hierarchical chunks with metadata.",
    )
    parser.add_argument(
        "sections_json",
        help="Path to the sections JSON file (output of extract.py).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON path.  Defaults to <input_stem>_chunks.json",
    )
    parser.add_argument(
        "--resume-id",
        default=None,
        help="Unique resume identifier.  Defaults to input filename stem.",
    )
    parser.add_argument(
        "--person-name",
        default=None,
        help="Candidate name (auto-detected from Personal Info if omitted).",
    )
    args = parser.parse_args()

    # Load sections
    sections = json.loads(
        Path(args.sections_json).read_text(encoding="utf-8")
    )

    resume_id = args.resume_id or Path(args.sections_json).stem
    output_path = args.output or f"{Path(args.sections_json).stem}_chunks.json"

    # Build chunks
    chunks = build_chunks(sections, resume_id, args.person_name)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # ── Pretty summary ─────────────────────────────────────────────────
    section_chunks = [c for c in chunks if c["level"] == "section"]
    sub_chunks = [c for c in chunks if c["level"] == "sub_item"]

    print(f"\n✅ Generated {len(chunks)} total chunks:")
    print(f"   📁 {len(section_chunks)} section-level chunks")
    print(f"   📄 {len(sub_chunks)} sub-item chunks\n")

    for c in chunks:
        level_icon = "📁" if c["level"] == "section" else "  📄"
        label = c["section"]
        if c["sub_section"]:
            label += f" → {c['sub_section']}"
        preview = c["content"][:60].replace("\n", " ")
        print(f"   {level_icon} {label}: {preview}...")

    print(f"\n   💾 Saved → {output_path}\n")


if __name__ == "__main__":
    main()
