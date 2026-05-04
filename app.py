"""
SkillLens — FastAPI Web Application
====================================
Resume upload + RAG-powered resume matching using Endee vector DB
and Google Generative AI.

Run:  uvicorn app:app --host 0.0.0.0 --port 5001 --reload
"""

import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from search_engine.build_collection import collection_init
from agent import execute

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()

app = FastAPI(title="SkillLens", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = BASE_DIR / "templates"

DEFAULT_COLLECTION = "skilllens_resumes"


@app.get("/")
async def index():
    """Serve the frontend."""
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


@app.get("/files/{filename:path}")
async def serve_file(filename: str):
    """Serve an uploaded resume file."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))


@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    collection: str = Form(default=""),
):
    """
    Upload one or more resume files (PDF/DOCX).
    Form fields:
      - files: one or more files
      - collection: (optional) collection/index name
    """
    try:
        if not collection.strip():
            collection = DEFAULT_COLLECTION
        # Sanitize: replace spaces and special chars
        collection = collection.strip().replace(" ", "_").lower()

        saved_paths = []
        for f in files:
            if not f.filename:
                continue
            ext = Path(f.filename).suffix.lower()
            if ext not in (".pdf", ".docx", ".doc"):
                continue
            dest = UPLOAD_DIR / f.filename
            content = await f.read()
            with open(str(dest), "wb") as out:
                out.write(content)
            saved_paths.append(str(dest))

        if not saved_paths:
            raise HTTPException(
                status_code=400, detail="No valid PDF/DOCX files found"
            )

        # Run the ingestion pipeline in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, collection_init, saved_paths, collection)

        return {
            "status": "success",
            "message": f"Ingested {len(saved_paths)} resume(s) into '{collection}'",
            "files": [Path(p).name for p in saved_paths],
            "collection": collection,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
async def list_collections():
    """List all existing Endee indexes."""
    try:
        from endee import Endee

        client = Endee()
        client.set_base_url("http://127.0.0.1:8080/api/v1")
        result = client.list_indexes()
        index_list = (
            result.get("indexes", []) if isinstance(result, dict) else result
        )
        names = []
        for idx in index_list:
            if isinstance(idx, dict):
                name = idx.get("name", "")
            else:
                name = str(idx)
            if name:
                names.append(name)
        return {"collections": names}
    except Exception as e:
        return {"collections": [], "error": str(e)}


@app.delete("/collections/{name}")
async def delete_collection(name: str):
    """Delete an Endee index by name."""
    try:
        from endee import Endee

        client = Endee()
        client.set_base_url("http://127.0.0.1:8080/api/v1")
        client.delete_index(name)
        return {"status": "success", "message": f"Collection '{name}' deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query")
async def query(
    q: str = Query(..., description="The user query"),
    collection: str = Query(default="skilllens_resumes", description="Collection name"),
    limit: int = Query(default=3, description="Number of results"),
):
    """
    Query endpoint — returns ranked resumes with normalized scores.
    No streaming, no LLM response — just resumes and their scores.

    Query params:
      - q: the user query
      - collection: (optional) collection name
      - limit: (optional) number of results, default 3
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Missing query parameter 'q'")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, execute, q.strip(), collection.strip(), limit
    )

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
