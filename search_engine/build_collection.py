"""
End-to-end: PDFs → extract → chunk → embed → Endee
Make sure Endee is running: ./run.sh (from the project root)
"""

from search_engine.search_engine import SearchEngine
from chunking.chunker import build_chunks
from chunking.extract import extract_text, detect_sections
from pathlib import Path


def ingest_resumes(engine, resume_paths):
    """Extract → chunk → push to Endee for each resume."""
    all_chunks = []

    for pdf_path in resume_paths:
        if not Path(pdf_path).exists():
            print(f"  ⚠️  Skipping {pdf_path} (not found)")
            continue

        stem = Path(pdf_path).stem
        print(f"\n  📄 Processing: {pdf_path}")

        # Step 1: Extract text from PDF
        raw_text = extract_text(pdf_path)
        print(f"     Extracted {len(raw_text):,} chars")

        # Step 2: Detect sections
        sections = detect_sections(raw_text)
        print(f"     Detected {len(sections)} sections")

        # Step 3: Build hierarchical chunks with metadata
        chunks = build_chunks(sections, resume_id=stem)
        section_count = sum(1 for c in chunks if c["level"] == "section")
        sub_count = sum(1 for c in chunks if c["level"] == "sub_item")
        print(f"     Generated {len(chunks)} chunks ({section_count} sections, {sub_count} sub-items)")

        all_chunks.extend(chunks)

    # Step 4: Push all chunks to Endee
    print(f"\n  Pushing {len(all_chunks)} total chunks to Endee...")
    engine._push_points(all_chunks)
    print("  ✅ Ingestion complete!")
    return all_chunks


def collection_init(resume_paths, collection_name):
    """Create an Endee index and ingest resumes into it."""
    engine = SearchEngine()
    print("-" * 60)
    result = engine._create_collection(collection_name)
    if result == "fail":
        print(f"Collection '{collection_name}' already exists, reusing it.")
        engine.collection_name = collection_name
    else:
        print(f"Created collection: {collection_name}")

    # Ingest all resumes (PDF → chunks → Endee)
    print("\nIngesting resumes...")
    ingest_resumes(engine, resume_paths)
