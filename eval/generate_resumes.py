"""
Synthetic Resume Generator for SkillLens Evaluation
=====================================================
Generates 100 synthetic resumes (10 roles × 10 each) using Llama 3.1
via Ollama running on WSL.

Each resume is generated as PLAIN TEXT with standard section headers
(Experience, Technical Skills, Projects, etc.) so that the existing
SkillLens pipeline (detect_sections → build_chunks → embed) processes
them exactly as it would a real extracted resume.

Resumes are saved as .txt files in eval/resumes/.

Usage:
    python -m eval.generate_resumes
    python -m eval.generate_resumes --model llama3.1:8b
"""

import argparse
import json
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# 10 Job Roles
# ---------------------------------------------------------------------------

JOB_ROLES = [
    {
        "role": "Machine Learning Engineer",
        "slug": "ml_engineer",
        "skills": "Python, PyTorch, TensorFlow, Scikit-learn, NLP, Computer Vision, MLOps, "
                  "Hugging Face, ONNX, MLflow, Docker, Kubernetes, NumPy, Pandas",
        "domain": "artificial intelligence and machine learning",
    },
    {
        "role": "Backend Engineer",
        "slug": "backend_engineer",
        "skills": "Python, Go, Java, REST APIs, gRPC, PostgreSQL, MySQL, Redis, Kafka, "
                  "Docker, Microservices, FastAPI, Django, Spring Boot, System Design",
        "domain": "backend systems and server-side development",
    },
    {
        "role": "Frontend Engineer",
        "slug": "frontend_engineer",
        "skills": "React, TypeScript, JavaScript, HTML, CSS, Next.js, Vue.js, Tailwind CSS, "
                  "Webpack, Vite, Figma, Jest, Cypress, Redux, GraphQL",
        "domain": "frontend web development and user interfaces",
    },
    {
        "role": "DevOps Engineer",
        "slug": "devops_engineer",
        "skills": "Kubernetes, Docker, Terraform, Ansible, AWS, GCP, Azure, CI/CD, Jenkins, "
                  "GitHub Actions, Prometheus, Grafana, Linux, Bash, Helm",
        "domain": "DevOps, cloud infrastructure, and site reliability",
    },
    {
        "role": "Data Scientist",
        "slug": "data_scientist",
        "skills": "Python, R, SQL, Statistics, Pandas, NumPy, Scikit-learn, Matplotlib, "
                  "Jupyter, A/B Testing, Regression, Classification, Tableau",
        "domain": "data science, statistical analysis, and predictive modeling",
    },
    {
        "role": "Cybersecurity Analyst",
        "slug": "cybersecurity_analyst",
        "skills": "Penetration Testing, SIEM, Splunk, Firewalls, IDS/IPS, SOC Operations, "
                  "NIST, ISO 27001, Incident Response, Wireshark, Metasploit, Burp Suite",
        "domain": "cybersecurity, threat analysis, and information security",
    },
    {
        "role": "Mobile App Developer",
        "slug": "mobile_developer",
        "skills": "React Native, Flutter, Swift, Kotlin, Dart, iOS, Android, Firebase, "
                  "SQLite, Realm, Xcode, Android Studio, Redux, State Management",
        "domain": "mobile application development for iOS and Android",
    },
    {
        "role": "Cloud Architect",
        "slug": "cloud_architect",
        "skills": "AWS, Azure, GCP, Microservices, Serverless, Lambda, CloudFormation, "
                  "Terraform, VPC, IAM, S3, EC2, EKS, DynamoDB, Cloud Security",
        "domain": "cloud architecture, infrastructure design, and cloud-native solutions",
    },
    {
        "role": "Full Stack Engineer",
        "slug": "fullstack_engineer",
        "skills": "Node.js, React, TypeScript, MongoDB, PostgreSQL, GraphQL, REST APIs, "
                  "Docker, Express.js, Next.js, Tailwind CSS, Redis, Git, CI/CD",
        "domain": "full-stack web development with both frontend and backend",
    },
    {
        "role": "Data Engineer",
        "slug": "data_engineer",
        "skills": "Apache Spark, Kafka, Airflow, Snowflake, dbt, SQL, Python, AWS Glue, "
                  "Redshift, BigQuery, ETL/ELT, Data Modeling, Delta Lake, Databricks",
        "domain": "data engineering, ETL pipelines, and data infrastructure",
    },
]

# ---------------------------------------------------------------------------
# Prompt — tells the model to produce plain text with section headers
# that match what detect_sections() recognizes
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """Generate a realistic professional resume for a {role} with {experience_years} years of experience.

Write it as PLAIN TEXT with these exact section headers on their own line:

Summary
Technical Skills
Experience
Projects
Education
Certifications

RULES:
- First line must be the candidate's full name (unique, realistic).
- Second line: email address.
- Third line: linkedin URL.
- Fourth line: City, State.
- Then a blank line before "Summary".
- Under "Experience", format each job as:
  Job Title at Company Name | Tech1, Tech2, Tech3  StartYear-EndYear
  – Achievement bullet with metrics
  – Another bullet with technical details
  – Third bullet about impact
  Include 2-3 job entries.
- Under "Projects", format each project as:
  Project Name | Tech1, Tech2  Year
  – What was built
  – Technical details and results
  Include 2-3 project entries.
- Under "Technical Skills", list skills by category:
  Languages: Python, Go
  Frameworks: PyTorch, FastAPI
  Tools: Docker, AWS
- Use these technologies prominently: {skills}
- This is resume #{resume_index} for this role — use a UNIQUE name, different companies, and different project topics.
- Do NOT use markdown formatting (no **, no ##, no ```). Just plain text with section headers.
- Write ONLY the resume text. No commentary before or after."""


def call_ollama(prompt: str, model: str, base_url: str) -> str | None:
    """Call Ollama API directly via HTTP."""
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "num_predict": 2048,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama API error: {e}")
        return None


def generate_single_resume(model: str, base_url: str, role_info: dict, resume_index: int) -> str | None:
    """Generate one plain-text resume."""
    exp_pool = [1, 2, 3, 4, 5, 6, 7, 8, 3, 5]
    experience_years = exp_pool[(resume_index - 1) % len(exp_pool)]

    prompt = PROMPT_TEMPLATE.format(
        role=role_info["role"],
        skills=role_info["skills"],
        experience_years=experience_years,
        resume_index=resume_index,
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        text = call_ollama(prompt, model, base_url)
        if text and len(text) > 200:
            # Basic validation: should contain at least some section headers
            text_lower = text.lower()
            has_experience = "experience" in text_lower
            has_skills = "skill" in text_lower
            if has_experience and has_skills:
                return text
            else:
                print(f"      Attempt {attempt}/{max_retries}: missing sections, retrying...")
        else:
            print(f"      Attempt {attempt}/{max_retries}: response too short, retrying...")

        if attempt < max_retries:
            time.sleep(2)

    return None


def generate_all_resumes(
    model_name: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    output_dir: str = None,
):
    """Generate 100 resumes (10 roles × 10) and save as .txt files."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "resumes")

    resumes_dir = Path(output_dir)
    resumes_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": model_name,
        "total_resumes": 0,
        "roles": {},
        "resumes": [],
    }

    total = len(JOB_ROLES) * 10
    generated = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"  SkillLens Resume Generator")
    print(f"  Model: {model_name} @ {base_url}")
    print(f"  Target: {total} resumes ({len(JOB_ROLES)} roles × 10)")
    print(f"  Output: {resumes_dir}")
    print(f"{'='*60}\n")

    for role_info in JOB_ROLES:
        role_name = role_info["role"]
        role_slug = role_info["slug"]
        role_files = []

        print(f"\n📋 Role: {role_name}")

        for i in range(1, 11):
            filename = f"{role_slug}_{i:02d}.txt"
            filepath = resumes_dir / filename

            # Skip if already generated
            if filepath.exists():
                print(f"   ✅ [{i:2d}/10] {filename} (exists, skipping)")
                role_files.append(filename)
                generated += 1
                manifest["resumes"].append({
                    "filename": filename,
                    "role": role_name,
                    "slug": role_slug,
                    "resume_id": filepath.stem,
                })
                continue

            print(f"   ⏳ [{i:2d}/10] Generating {filename}...", end=" ", flush=True)
            start_time = time.time()

            text = generate_single_resume(model_name, base_url, role_info, i)
            elapsed = time.time() - start_time

            if text:
                filepath.write_text(text, encoding="utf-8")
                role_files.append(filename)
                generated += 1

                # Get name from first line
                person_name = text.split("\n")[0].strip()
                print(f"✅ ({elapsed:.1f}s) — {person_name}")

                manifest["resumes"].append({
                    "filename": filename,
                    "role": role_name,
                    "slug": role_slug,
                    "resume_id": filepath.stem,
                    "person_name": person_name,
                })
            else:
                failed += 1
                print(f"❌ FAILED ({elapsed:.1f}s)")

        manifest["roles"][role_slug] = {
            "role": role_name,
            "files": role_files,
            "count": len(role_files),
        }
        print(f"   → {len(role_files)}/10 done")

    manifest["total_resumes"] = generated

    manifest_path = resumes_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  ✅ Generated: {generated}/{total}")
    print(f"  ❌ Failed:    {failed}/{total}")
    print(f"  📁 Saved to:  {resumes_dir}")
    print(f"{'='*60}\n")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic resumes.")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    generate_all_resumes(model_name=args.model, base_url=args.base_url, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
