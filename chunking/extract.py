"""
Resume Section-wise Extractor
==============================
Extracts resume content section-by-section and saves it as structured JSON.
Supports PDF (including multi-column layouts) and DOCX formats.

Uses PyMuPDF (fitz) for PDF extraction — text blocks are extracted with
bounding boxes, clustered into columns by x-position, and each column is
processed top-to-bottom before section detection runs.

Usage:
    python extract.py resume.pdf -o sections.json
    python extract.py resume.docx
"""

import argparse
import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Canonical section headers (lowercase) — covers most standard resume formats
# ---------------------------------------------------------------------------
SECTION_HEADERS = [
    "contact information",
    "contact",
    "personal information",
    "personal details",
    "summary",
    "professional summary",
    "executive summary",
    "profile",
    "about me",
    "about",
    "objective",
    "career objective",
    "education",
    "academic background",
    "qualifications",
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "work history",
    "internships",
    "internship experience",
    "skills",
    "technical skills",
    "core competencies",
    "competencies",
    "key skills",
    "projects",
    "academic projects",
    "personal projects",
    "certifications",
    "certificates",
    "licenses",
    "awards",
    "awards and honors",
    "honors",
    "achievements",
    "accomplishments",
    "publications",
    "research",
    "research experience",
    "languages",
    "interests",
    "hobbies",
    "hobbies and interests",
    "extracurricular activities",
    "activities",
    "volunteer experience",
    "volunteer",
    "volunteering",
    "references",
    "training",
    "professional development",
    "courses",
    "coursework",
    "relevant coursework",
    "leadership",
    "leadership experience",
    "organizations",
    "memberships",
    "professional memberships",
    "affiliations",
    "professional affiliations",
    "declaration",
]


# ---------------------------------------------------------------------------
# Column-aware PDF text extraction (PyMuPDF)
# ---------------------------------------------------------------------------

def _cluster_blocks_into_columns(
    blocks: list[dict], gap_ratio: float = 0.35
) -> list[list[dict]]:
    """
    Cluster text blocks into columns based on their x0 (left edge) positions.

    Algorithm:
      1. Collect all unique x0 values from the blocks.
      2. Sort them and find "gaps" — if the distance between consecutive x0
         values is larger than `gap_ratio * page_width`, it marks a column
         boundary.
      3. Assign each block to the column whose x-range contains its x0.

    This handles 1-column, 2-column, sidebar, and asymmetric layouts.
    """
    if not blocks:
        return []

    # Determine page width from the blocks themselves
    all_x0 = sorted(set(b["x0"] for b in blocks))
    all_x1 = [b["x1"] for b in blocks]
    page_width = max(all_x1) - min(all_x0) if all_x1 else 1
    gap_threshold = page_width * gap_ratio

    # Find column boundaries by detecting large gaps in x0 values
    column_boundaries = [all_x0[0]]
    for i in range(1, len(all_x0)):
        if all_x0[i] - all_x0[i - 1] > gap_threshold:
            column_boundaries.append(all_x0[i])

    # Assign blocks to columns
    columns: list[list[dict]] = [[] for _ in column_boundaries]
    for block in blocks:
        # Find the closest column boundary ≤ block's x0
        col_idx = 0
        for i, boundary in enumerate(column_boundaries):
            if block["x0"] >= boundary - 5:  # small tolerance
                col_idx = i
        columns[col_idx].append(block)

    # Sort each column's blocks top-to-bottom
    for col in columns:
        col.sort(key=lambda b: (b["y0"], b["x0"]))

    # Remove empty columns
    return [col for col in columns if col]


def extract_text_from_pdf(filepath: str) -> str:
    """
    Extract text from a PDF with column-aware ordering.

    For each page:
      1. Extract text blocks with bounding boxes via PyMuPDF.
      2. Cluster blocks into columns by x-position.
      3. Read columns left-to-right, blocks top-to-bottom within each column.

    This ensures multi-column resumes are read in the correct order.
    """
    doc = fitz.open(filepath)
    page_texts: list[str] = []

    for page in doc:
        # Extract text blocks: each is (x0, y0, x1, y1, "text", block_no, block_type)
        raw_blocks = page.get_text("blocks")
        text_blocks = []
        for b in raw_blocks:
            # block_type 0 = text, 1 = image
            if b[6] == 0 and b[4].strip():
                text_blocks.append({
                    "x0": b[0],
                    "y0": b[1],
                    "x1": b[2],
                    "y1": b[3],
                    "text": b[4].strip(),
                })

        if not text_blocks:
            continue

        columns = _cluster_blocks_into_columns(text_blocks)

        # Read columns left-to-right
        column_texts = []
        for col_blocks in columns:
            col_text = "\n".join(b["text"] for b in col_blocks)
            column_texts.append(col_text)

        page_texts.append("\n".join(column_texts))

    doc.close()
    return "\n".join(page_texts)


# ---------------------------------------------------------------------------
# DOCX text extraction
# ---------------------------------------------------------------------------

def extract_text_from_docx(filepath: str) -> str:
    """Extract raw text from a DOCX file using python-docx."""
    from docx import Document
    doc = Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text(filepath: str) -> str:
    """Route to the correct extractor based on file extension."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(filepath)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported: .pdf, .docx"
        )


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _build_header_pattern() -> re.Pattern:
    """
    Build a compiled regex that matches any known section header at the
    start of a line.  Handles optional trailing colons, dashes, pipes, and
    common decorators like ── or ===.
    """
    escaped = [re.escape(h) for h in SECTION_HEADERS]
    escaped.sort(key=len, reverse=True)
    alternatives = "|".join(escaped)

    pattern = (
        r"^[ \t]*[#\-=─►•|]*[ \t]*"
        rf"({alternatives})"
        r"[ \t]*[:—\-─=|]*[ \t]*$"
    )
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


HEADER_RE = _build_header_pattern()


def _normalize_section_name(raw: str) -> str:
    """Title-case a matched header string for clean JSON output."""
    return raw.strip().title()


def detect_sections(text: str) -> list[dict]:
    """
    Detect section boundaries in resume text and return a list of
    {"section": <name>, "content": <text>} dicts.

    Lines before the first detected section header are captured under
    "Personal Info" (typically name, contact details, links).
    """
    lines = text.split("\n")
    sections: list[dict] = []
    current_section = None
    current_lines: list[str] = []

    for line in lines:
        match = HEADER_RE.match(line)
        if match:
            content = "\n".join(current_lines).strip()
            if current_section is None and content:
                sections.append({
                    "section": "Personal Info",
                    "content": content,
                })
            elif current_section is not None:
                sections.append({
                    "section": current_section,
                    "content": content,
                })
            current_section = _normalize_section_name(match.group(1))
            current_lines = []
        else:
            current_lines.append(line)

    # Last section
    if current_section is not None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({
                "section": current_section,
                "content": content,
            })
    elif current_lines:
        sections.append({
            "section": "Full Resume",
            "content": "\n".join(current_lines).strip(),
        })

    # ------------------------------------------------------------------
    # Post-processing: merge empty sections with the next section.
    # If "Technical Skills" has no content and is followed by "Languages"
    # with content, it means "Languages" was a sub-heading, not a
    # top-level section.  Merge: keep "Technical Skills" as the name,
    # use the next section's content.
    # ------------------------------------------------------------------
    merged: list[dict] = []
    i = 0
    while i < len(sections):
        sec = sections[i]
        if not sec["content"] and i + 1 < len(sections):
            # Empty section → merge with the next one, keep this name
            next_sec = sections[i + 1]
            merged.append({
                "section": sec["section"],
                "content": next_sec["content"],
            })
            i += 2  # skip the next section (absorbed)
        else:
            merged.append(sec)
            i += 1

    return merged


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_to_json(sections: list[dict], output_path: str) -> None:
    """Write sections list to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(sections)} sections → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from chunker import build_chunks

    parser = argparse.ArgumentParser(
        description="Extract resume into hierarchical chunks (JSON).",
    )
    parser.add_argument(
        "resume",
        help="Path to the resume file (PDF or DOCX).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON path. Defaults to <resume_stem>_chunks.json",
    )
    parser.add_argument(
        "--resume-id",
        default=None,
        help="Unique resume identifier. Defaults to filename stem.",
    )
    parser.add_argument(
        "--person-name",
        default=None,
        help="Candidate name (auto-detected from Personal Info if omitted).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.resume):
        parser.error(f"File not found: {args.resume}")

    stem = Path(args.resume).stem
    resume_id = args.resume_id or stem
    output_path = args.output or f"{stem}_chunks.json"

    # Step 1: Extract text → detect sections
    print(f"📄 Reading: {args.resume}")
    raw_text = extract_text(args.resume)
    print(f"   Extracted {len(raw_text):,} characters of text.")

    sections = detect_sections(raw_text)
    print(f"   Detected {len(sections)} section(s):")
    for sec in sections:
        preview = sec["content"][:80].replace("\n", " ")
        print(f"     • {sec['section']}: {preview}...")

    # Step 2: Build hierarchical chunks with metadata
    chunks = build_chunks(sections, resume_id, args.person_name)

    # Step 3: Save
    save_to_json(chunks, output_path)

    section_chunks = [c for c in chunks if c["level"] == "section"]
    sub_chunks = [c for c in chunks if c["level"] == "sub_item"]
    print(f"\n   📁 {len(section_chunks)} section-level chunks")
    print(f"   📄 {len(sub_chunks)} sub-item chunks")


if __name__ == "__main__":
    main()
