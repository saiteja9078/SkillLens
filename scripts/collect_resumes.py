"""
Recursively find all resume files (PDF, DOCX) in a directory.

Usage:
    python collect_resumes.py /path/to/resumes
    python collect_resumes.py /path/to/resumes --extensions pdf docx
"""

import argparse
from pathlib import Path


def collect_resumes(directory, extensions=("pdf", "docx", "doc")):
    """Recursively find all resume files in a directory."""
    directory = Path(directory)
    resumes = []

    for ext in extensions:
        resumes.extend(directory.rglob(f"*.{ext}"))

    # Sort for consistent ordering
    resumes = sorted(resumes)
    return [str(r) for r in resumes]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find all resume files in a directory.")
    parser.add_argument("directory", help="Root directory to search")
    parser.add_argument("--extensions", nargs="+", default=["pdf", "docx", "doc"],
                        help="File extensions to look for (default: pdf docx doc)")
    args = parser.parse_args()

    resumes = collect_resumes(args.directory, args.extensions)

    print(f"Found {len(resumes)} resume(s):\n")
    for r in resumes:
        print(f"  {r}")
