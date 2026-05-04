"""
RAG Agent — Google Generative AI + Endee
=========================================
Prompt enhancement → Endee hybrid search → resume ranking with normalized scores.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from search_engine.search_engine import SearchEngine

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

guidelines = """You are a prompt enhancement agent operating inside a Retrieval-Augmented Generation (RAG) pipeline for resume retrieval.
Your task is to:
1. Rewrite and enhance the user's input into a retrieval-optimized query
2. Determine how many resumes the user wants returned

Constraints:
- You are given only one opportunity to enhance the prompt.
- You must NOT ask follow-up questions.
- You must NOT add assumptions that are not reasonably inferable from the user input.
- Do NOT generate answers, summaries, or explanations.

Enhancement Guidelines:
- Extract and explicitly state key skills, technologies, roles, experience level, domain, and constraints if present if not present then mention the skills / experience or anything that has to be need for user query.
- Normalize vague terms into concrete, searchable phrasing.
- Preserve the original intent and meaning of the user query.

Number Detection:
- If the user says "a candidate" or "one resume" -> count = 1
- If the user says "top 3" or "3 candidates" -> count = 3
- If the user says "find candidates" (plural, no number) -> count = 3
- If the user says "all matching" or "all resumes" -> count = 10
- Default to 3 if unclear.

Output Format:
- Return ONLY a valid JSON object, nothing else:
  {{"query": "the enhanced retrieval query", "count": 3}}
- No markdown, no code blocks, no commentary."""


def rank_resumes(sources, top_k_chunks=3, top_n_resumes=5):
    """
    Rank resumes by top-k mean similarity.

    For each resume:
      1. Collect all chunk scores
      2. Sort descending, take top `top_k_chunks`
      3. Compute mean of those top scores

    Returns:
      - Sorted list of (resume_id, mean_score, chunk_count, person_name)
    """
    from collections import defaultdict

    resume_chunks = defaultdict(list)
    for s in sources:
        rid = s.get("resume_id", "")
        if rid:
            resume_chunks[rid].append(s)

    # Compute top-k mean per resume
    resume_scores = []
    for rid, chunks in resume_chunks.items():
        scores = sorted(
            [c["similarity"] for c in chunks if isinstance(c["similarity"], (int, float))],
            reverse=True,
        )
        top_scores = scores[:top_k_chunks]
        if top_scores:
            mean_score = sum(top_scores) / len(top_scores)
        else:
            mean_score = 0.0

        # Get person name from the first chunk
        person_name = chunks[0].get("person_name", "Unknown") if chunks else "Unknown"
        resume_scores.append((rid, mean_score, len(chunks), person_name))

    # Sort by mean score descending
    resume_scores.sort(key=lambda x: x[1], reverse=True)

    # Take top N resumes
    return resume_scores[:top_n_resumes]


def normalize_scores(resume_scores):
    """
    Min-max normalize scores to [0, 1] range.
    If all scores are equal, all get 1.0.
    """
    if not resume_scores:
        return []

    raw_scores = [s[1] for s in resume_scores]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    score_range = max_score - min_score

    normalized = []
    for rid, score, count, person_name in resume_scores:
        if score_range == 0:
            norm_score = 1.0
        else:
            norm_score = (score - min_score) / score_range
        normalized.append((rid, norm_score, count, person_name))

    return normalized


def execute(prompt: str, collection_name: str, top_n_resumes: int = 3):
    """
    RAG pipeline — returns resumes with matching scores.

    Pipeline:
      1. Enhance user prompt + extract desired count
      2. Retrieve ALL chunks from Endee (single query)
      3. Group by resume_id, compute top-3 mean score per resume
      4. Return ranked resumes with scores

    Returns:
      dict with:
        - query: the enhanced query used
        - resumes: list of {resume_id, person_name, score, chunks_matched, filename}
    """
    import json as _json
    from pathlib import Path as _Path

    # Step 1: Enhance the prompt + extract desired count
    enhance_temp = ChatPromptTemplate.from_messages(
        [
            ("system", guidelines),
            ("user", "prompt: {prompt}"),
        ]
    )
    enhance_resp = model.invoke(
        enhance_temp.format_messages(prompt=prompt)
    ).content.strip()

    # Parse JSON response
    enhanced_prompt = prompt  # fallback
    try:
        json_start = enhance_resp.find("{")
        json_end = enhance_resp.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = _json.loads(enhance_resp[json_start:json_end])
            enhanced_prompt = parsed.get("query", prompt)
            top_n_resumes = parsed.get("count", top_n_resumes)
    except Exception:
        enhanced_prompt = enhance_resp

    # Step 2: Retrieve ALL chunks from the collection
    engine = SearchEngine(collection_name=collection_name)
    total = engine.get_total_elements()
    # total_elements can be stale after recent inserts, so always attempt search
    search_limit = max(total, 50)

    try:
        results = engine.hybrid_search(enhanced_prompt, limit=search_limit)
    except Exception as e:
        return {
            "query": enhanced_prompt,
            "resumes": [],
            "error": f"Search failed on '{collection_name}': {str(e)}",
        }

    if not results:
        return {
            "query": enhanced_prompt,
            "resumes": [],
            "message": f"No results found in collection '{collection_name}'.",
        }

    # Build source list with scores
    sources = []
    for item in results:
        meta = item.get("meta", {})
        sources.append({
            "person_name": meta.get("person_name", "Unknown"),
            "resume_id": meta.get("resume_id", ""),
            "similarity": item.get("similarity", 0.0),
        })

    # Step 3: Rank resumes by top-3 mean similarity
    resume_scores = rank_resumes(
        sources, top_k_chunks=3, top_n_resumes=top_n_resumes
    )

    # Step 4: Build response with resume file info
    uploads_dir = _Path(__file__).parent / "uploads"
    resumes = []
    for rid, score, count, person_name in resume_scores:
        # Find actual filename on disk
        actual_filename = rid + ".pdf"
        if uploads_dir.exists():
            for f in uploads_dir.iterdir():
                if f.stem == rid:
                    actual_filename = f.name
                    break

        resumes.append({
            "resume_id": rid,
            "person_name": person_name,
            "score": round(score, 4),
            "chunks_matched": count,
            "filename": actual_filename,
        })

    return {
        "query": enhanced_prompt,
        "resumes": resumes,
    }