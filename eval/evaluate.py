"""
SkillLens Evaluation Pipeline
===============================
Loads plain-text synthetic resumes, runs them through the full SkillLens
pipeline (detect_sections → build_chunks → embed → Endee), then evaluates
retrieval quality using LLM-as-a-judge.

Usage:
    python -m eval.evaluate
    python -m eval.evaluate --model llama3.1:8b --top-k 10
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

# Add parent dir so we can import SkillLens modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking.extract import detect_sections
from chunking.chunker import build_chunks
from search_engine.search_engine import SearchEngine


EVAL_COLLECTION = "eval_skilllens"

# ---------------------------------------------------------------------------
# Job description queries — one per role
# ---------------------------------------------------------------------------

ROLE_QUERIES = {
    "ml_engineer": (
        "Looking for a Machine Learning Engineer with strong experience in "
        "PyTorch, TensorFlow, deep learning, NLP, computer vision, and MLOps. "
        "Should have hands-on experience building and deploying ML models."
    ),
    "backend_engineer": (
        "Looking for a Backend Engineer skilled in Python, Go, REST APIs, "
        "PostgreSQL, Redis, Docker, microservices architecture, and system design."
    ),
    "frontend_engineer": (
        "Looking for a Frontend Engineer with expertise in React, TypeScript, "
        "Next.js, CSS, responsive design, and modern frontend tooling."
    ),
    "devops_engineer": (
        "Looking for a DevOps Engineer experienced with Kubernetes, Docker, "
        "Terraform, CI/CD pipelines, AWS, monitoring, and infrastructure as code."
    ),
    "data_scientist": (
        "Looking for a Data Scientist with strong skills in Python, statistics, "
        "Pandas, Scikit-learn, A/B testing, data visualization, and machine learning."
    ),
    "cybersecurity_analyst": (
        "Looking for a Cybersecurity Analyst with experience in penetration testing, "
        "SIEM tools, incident response, vulnerability assessment, and compliance."
    ),
    "mobile_developer": (
        "Looking for a Mobile App Developer skilled in React Native, Flutter, "
        "Swift, Kotlin, Firebase, and cross-platform mobile development."
    ),
    "cloud_architect": (
        "Looking for a Cloud Architect with expertise in AWS, Azure, GCP, "
        "serverless architecture, microservices, and cloud security."
    ),
    "fullstack_engineer": (
        "Looking for a Full Stack Engineer with experience in Node.js, React, "
        "TypeScript, MongoDB, PostgreSQL, GraphQL, and Docker."
    ),
    "data_engineer": (
        "Looking for a Data Engineer experienced with Apache Spark, Kafka, "
        "Airflow, Snowflake, dbt, SQL, and building data pipelines at scale."
    ),
}


# ---------------------------------------------------------------------------
# LLM Judge via Ollama API
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert hiring manager. Given a JOB DESCRIPTION and a RESUME, score relevance on a 0-3 scale:

3 = Highly Relevant: right role, right skills, strong fit
2 = Relevant: related role, most key skills present
1 = Partially Relevant: some overlap, different specialization
0 = Not Relevant: little to no relevance

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_content}

Respond with ONLY a JSON object: {{"score": <0-3>, "reason": "<one sentence>"}}"""


def call_ollama(prompt: str, model: str, base_url: str, temperature: float = 0.0) -> str:
    """Call Ollama API directly."""
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 256},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def judge_resume(model: str, base_url: str, job_description: str, resume_text: str) -> dict:
    """Score a resume's relevance using LLM judge."""
    prompt = JUDGE_PROMPT.format(
        job_description=job_description,
        resume_content=resume_text[:3000],
    )
    try:
        content = call_ollama(prompt, model, base_url)
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(content[start:end])
            score = int(parsed.get("score", 0))
            return {"score": min(max(score, 0), 3), "reason": parsed.get("reason", "")}
    except Exception as e:
        print(f"      Judge error: {e}")
    return {"score": 0, "reason": "Failed to parse"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_roles: list[str], target: str) -> float:
    if not retrieved_roles:
        return 0.0
    return sum(1 for r in retrieved_roles if r == target) / len(retrieved_roles)


def recall_at_k(retrieved_roles: list[str], target: str, total: int = 10) -> float:
    if total == 0:
        return 0.0
    return sum(1 for r in retrieved_roles if r == target) / total


def compute_ndcg(scores: list[float], k: int = 10) -> float:
    def dcg(s, k):
        return sum(v / math.log2(i + 2) for i, v in enumerate(s[:k]))
    actual = dcg(scores, k)
    ideal = dcg(sorted(scores, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_manifest(resumes_dir: Path) -> dict:
    path = resumes_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest at {path}. Run generate_resumes.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_resumes(resumes_dir: Path, manifest: dict, engine: SearchEngine) -> dict:
    """
    Load .txt resumes → detect_sections() → build_chunks() → push to Endee.
    Returns resume_id → role_slug mapping.
    """
    resume_role_map = {}
    all_chunks = []

    for entry in manifest["resumes"]:
        filepath = resumes_dir / entry["filename"]
        if not filepath.exists():
            print(f"  ⚠️  Skipping {entry['filename']} (not found)")
            continue

        resume_id = filepath.stem
        role_slug = entry["slug"]
        resume_role_map[resume_id] = role_slug

        # Read plain text
        raw_text = filepath.read_text(encoding="utf-8")

        # Run through the SAME pipeline as production
        sections = detect_sections(raw_text)
        chunks = build_chunks(sections, resume_id=resume_id)
        all_chunks.extend(chunks)

        sec_count = sum(1 for c in chunks if c["level"] == "section")
        sub_count = sum(1 for c in chunks if c["level"] == "sub_item")
        print(f"  📄 {resume_id}: {len(sections)} sections → {len(chunks)} chunks "
              f"({sec_count} section, {sub_count} sub-item)")

    print(f"\n  Pushing {len(all_chunks)} chunks to Endee...")
    engine._push_points(all_chunks)
    print(f"  ✅ Ingestion complete!")
    return resume_role_map


def retrieve_and_score(
    engine: SearchEngine,
    model: str,
    base_url: str,
    role_slug: str,
    job_description: str,
    resume_role_map: dict,
    resumes_dir: Path,
    top_k: int = 10,
) -> dict:
    """Retrieve top-k resumes and score with LLM judge."""

    # Retrieve ALL chunks then rank (same as agent.py)
    total = engine.get_total_elements()
    search_limit = max(total, 50)
    results = engine.hybrid_search(job_description, limit=search_limit)

    if not results:
        return {"error": "No results"}

    # Group by resume_id, compute top-3 mean (same as agent.py)
    resume_chunks = defaultdict(list)
    for item in results:
        meta = item.get("meta", {})
        rid = meta.get("resume_id", "")
        if rid:
            resume_chunks[rid].append({
                "similarity": item.get("similarity", 0.0),
                "person_name": meta.get("person_name", "Unknown"),
            })

    resume_scores = []
    for rid, chunks in resume_chunks.items():
        scores = sorted([c["similarity"] for c in chunks], reverse=True)
        top3 = scores[:3]
        mean = sum(top3) / len(top3) if top3 else 0.0
        name = chunks[0]["person_name"]
        resume_scores.append((rid, mean, len(chunks), name))

    resume_scores.sort(key=lambda x: x[1], reverse=True)
    top_resumes = resume_scores[:top_k]

    # Judge each retrieved resume
    retrieved_details = []
    retrieved_roles = []
    judge_scores = []

    for rid, sim, chunk_count, person_name in top_resumes:
        actual_role = resume_role_map.get(rid, "unknown")
        retrieved_roles.append(actual_role)

        # Load resume text for judging
        resume_text = ""
        for ext in (".txt", ".json"):
            fpath = resumes_dir / f"{rid}{ext}"
            if fpath.exists():
                resume_text = fpath.read_text(encoding="utf-8")
                break

        result = judge_resume(model, base_url, job_description, resume_text)
        judge_scores.append(result["score"])

        is_correct = actual_role == role_slug
        status = "✅" if is_correct else "❌"
        print(f"      {status} {rid} (sim={sim:.4f}, judge={result['score']}/3) — {actual_role}")

        retrieved_details.append({
            "resume_id": rid,
            "person_name": person_name,
            "actual_role": actual_role,
            "is_correct_role": is_correct,
            "similarity_score": round(sim, 4),
            "judge_score": result["score"],
            "judge_reason": result["reason"],
        })

    # Compute metrics
    prec = precision_at_k(retrieved_roles, role_slug)
    rec = recall_at_k(retrieved_roles, role_slug, total=10)
    mean_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
    norm_rel = mean_judge / 3.0
    ndcg = compute_ndcg([float(s) for s in judge_scores], k=top_k)
    overall = 0.3 * prec + 0.2 * rec + 0.3 * norm_rel + 0.2 * ndcg

    return {
        "role": role_slug,
        "query": job_description,
        "top_k": top_k,
        "retrieved": retrieved_details,
        "metrics": {
            "precision_at_k": round(prec, 4),
            "recall_at_k": round(rec, 4),
            "mean_judge_score": round(mean_judge, 4),
            "normalized_relevance": round(norm_rel, 4),
            "ndcg_at_k": round(ndcg, 4),
            "overall_score": round(overall, 4),
        },
    }


def run_evaluation(
    model_name: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    top_k: int = 10,
    resumes_dir: str = None,
    results_dir: str = None,
):
    eval_dir = Path(__file__).parent
    resumes_dir = Path(resumes_dir) if resumes_dir else eval_dir / "resumes"
    results_dir = Path(results_dir) if results_dir else eval_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SkillLens Evaluation Pipeline")
    print(f"  Judge: {model_name} @ {base_url}")
    print(f"  Top-K: {top_k}")
    print(f"{'='*60}\n")

    # Step 1: Load manifest
    print("📋 Step 1: Loading manifest...")
    manifest = load_manifest(resumes_dir)
    print(f"   Found {manifest['total_resumes']} resumes\n")

    # Step 2: Ingest via production pipeline
    print("📦 Step 2: Ingesting resumes (detect_sections → build_chunks → Endee)...")
    engine = SearchEngine()

    # Delete existing eval collection if present
    try:
        engine.client.delete_index(EVAL_COLLECTION)
        print(f"   Deleted existing '{EVAL_COLLECTION}'")
    except Exception:
        pass

    # Create index directly with correct SDK params
    try:
        engine.client.create_index(
            name=EVAL_COLLECTION,
            dimension=768,            # bge-base-en-v1.5 output dim
            space_type="cosine",
            sparse_model="default",   # enable hybrid (dense+sparse) search
        )
        print(f"   Created collection: {EVAL_COLLECTION}")
    except Exception as e:
        print(f"   Collection may already exist: {e}")

    engine.collection_name = EVAL_COLLECTION
    engine._index = engine.client.get_index(name=EVAL_COLLECTION)

    resume_role_map = ingest_resumes(resumes_dir, manifest, engine)
    print(f"   Ingested: {len(resume_role_map)} resumes\n")

    time.sleep(2)

    # Step 3: Evaluate each role
    print("🔍 Step 3: Retrieval + LLM-as-a-judge scoring...\n")

    all_results = {}
    all_metrics = []

    for role_slug, query in ROLE_QUERIES.items():
        role_name = manifest["roles"].get(role_slug, {}).get("role", role_slug)
        print(f"\n  📋 {role_name}")
        print(f"     Query: {query[:80]}...")

        result = retrieve_and_score(
            engine, model_name, base_url, role_slug, query,
            resume_role_map, resumes_dir, top_k,
        )
        all_results[role_slug] = result

        if "metrics" in result:
            all_metrics.append(result["metrics"])
            m = result["metrics"]
            print(f"     → P@{top_k}={m['precision_at_k']:.2f}  "
                  f"R@{top_k}={m['recall_at_k']:.2f}  "
                  f"Judge={m['mean_judge_score']:.2f}/3  "
                  f"NDCG={m['ndcg_at_k']:.2f}  "
                  f"Overall={m['overall_score']:.2f}")

    # Step 4: Aggregate + save
    if all_metrics:
        avg = {k: round(sum(m[k] for m in all_metrics) / len(all_metrics), 4) for k in all_metrics[0]}

        final = {
            "config": {"model": model_name, "top_k": top_k, "total_resumes": manifest["total_resumes"]},
            "per_role": all_results,
            "aggregate": avg,
        }

        results_path = results_dir / "eval_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)

        # Human-readable summary
        summary_path = results_dir / "eval_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"SkillLens Evaluation Results\n{'='*80}\n")
            f.write(f"Model: {model_name} | Top-K: {top_k} | Resumes: {manifest['total_resumes']}\n")
            f.write(f"{'='*80}\n\n")

            hdr = f"{'Role':<30} {'Prec@k':<10} {'Rec@k':<10} {'Judge':<12} {'NDCG':<10} {'Overall':<10}"
            f.write(hdr + "\n" + "-" * len(hdr) + "\n")

            for slug in ROLE_QUERIES:
                if slug in all_results and "metrics" in all_results[slug]:
                    m = all_results[slug]["metrics"]
                    name = manifest["roles"].get(slug, {}).get("role", slug)
                    f.write(f"{name:<30} {m['precision_at_k']:<10.4f} {m['recall_at_k']:<10.4f} "
                            f"{m['mean_judge_score']:<5.2f}/3.0   {m['ndcg_at_k']:<10.4f} "
                            f"{m['overall_score']:<10.4f}\n")

            f.write("-" * len(hdr) + "\n")
            f.write(f"{'AVERAGE':<30} {avg['precision_at_k']:<10.4f} {avg['recall_at_k']:<10.4f} "
                    f"{avg['mean_judge_score']:<5.2f}/3.0   {avg['ndcg_at_k']:<10.4f} "
                    f"{avg['overall_score']:<10.4f}\n")

        print(f"\n{'='*60}")
        print(f"  📊 Results: {results_path}")
        print(f"  📝 Summary: {summary_path}")
        print(f"  Precision@{top_k}: {avg['precision_at_k']:.4f}")
        print(f"  Recall@{top_k}:    {avg['recall_at_k']:.4f}")
        print(f"  Judge:        {avg['mean_judge_score']:.2f}/3.0")
        print(f"  NDCG@{top_k}:      {avg['ndcg_at_k']:.4f}")
        print(f"  Overall:      {avg['overall_score']:.4f}")
        print(f"{'='*60}\n")

    # Cleanup
    try:
        engine.client.delete_index(EVAL_COLLECTION)
        print(f"  🧹 Cleaned up '{EVAL_COLLECTION}'")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run SkillLens evaluation.")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model for judging")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--top-k", type=int, default=10, help="Resumes to retrieve per query")
    parser.add_argument("--resumes-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model, base_url=args.base_url, top_k=args.top_k,
        resumes_dir=args.resumes_dir, results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
