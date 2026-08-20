# SkillLens -- AI-Powered Resume Screening with Vector Search

## Project Overview

SkillLens is an AI-powered resume screening tool that lets recruiters upload resumes and search across them using natural language queries. It combines **semantic vector search** with **LLM-powered query enhancement** to find and rank the best-matching candidates for any role or skill set.

### Problem Statement

Manual resume screening is time-consuming and error-prone. Recruiters often sift through hundreds of resumes to find candidates matching specific skill requirements. Keyword-based search fails to capture semantic meaning -- a query for "machine learning experience" won't match resumes that mention "trained neural networks" or "built predictive models."

SkillLens solves this with a **Retrieval-Augmented Generation (RAG) pipeline** that:

1. Understands the _meaning_ behind queries, not just keywords
2. Retrieves all resume chunks and scores every resume using hybrid vector search
3. Ranks resumes by top-k mean similarity and lets the LLM control how many to return
4. Returns ranked resumes with their matching scores

### Use Case

**RAG (Retrieval-Augmented Generation)** for resume retrieval and ranking, where vector search is the core retrieval mechanism.

---

## System Design

### Architecture

```
User Query
    |
    v
[Prompt Enhancement] --- Gemini rewrites the query for retrieval
    |                     AND extracts desired resume count (k)
    v
[Full Collection Search]  Qdrant hybrid search across ALL chunks
    |                     (dense + sparse via RRF fusion)
    v
[Resume-Level Ranking] -- Group chunks by resume_id, compute
    |                     top-3 chunk mean score per resume
    v
[Top-K Selection] ------- Take top k resumes (k from user query)
    |                     Only these resumes are returned
    v
[JSON Response] --------- Returns ranked resumes with scores
                          and clickable PDF links in the UI
```

### Technical Stack

| Component         | Technology                                     |
| ----------------- | ---------------------------------------------- |
| Vector Database   | **Qdrant** (Docker, port 6333)                 |
| LLM               | Google Gemini 2.5 Flash (via LangChain)        |
| Dense Embeddings  | BAAI/bge-base-en-v1.5 (768-dim, via fastembed) |
| Sparse Embeddings | SPLADE++ (prithvida/Splade_PP_en_v1)           |
| Backend           | Python, FastAPI, Uvicorn                       |
| Frontend          | HTML, CSS, JavaScript (no frameworks)          |
| PDF Parsing       | PyMuPDF (fitz)                                 |
| Containerization  | Docker                                         |

### Project Structure

```
SkillLens/
|-- agent.py                    # RAG pipeline: prompt enhancement, ranking, scoring
|-- app.py                      # FastAPI web server with REST endpoints
|-- requirements.txt            # Python dependencies
|-- .env                        # Environment variables (GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY)
|-- templates/
|   |-- index.html              # Frontend: chat interface, upload, collection management
|-- search_engine/
|   |-- search_engine.py        # Qdrant client: collection creation, upsert, hybrid search
|   |-- build_collection.py     # Ingestion pipeline: PDF -> chunks -> Qdrant
|-- chunking/
|   |-- extract.py              # PDF/DOCX text extraction and section detection
|   |-- chunker.py              # Hierarchical chunking with metadata
|-- uploads/                    # Uploaded resume files (gitignored)
|-- qdrant_storage/             # Qdrant persistent data (gitignored, Docker volume)
```

---

## How Qdrant Is Used

[Qdrant](https://qdrant.tech) serves as the **vector database** at the core of the retrieval pipeline. It runs in Docker and is accessed via the `qdrant-client` Python SDK. Here is how it is integrated:

### 1. Collection Creation

When resumes are uploaded, SkillLens creates a Qdrant collection configured for **hybrid search** using named vectors (dense + sparse):

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, SparseIndexParams, Distance

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="skilllens_resumes",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
    },
)
```

### 2. Document Ingestion

Each resume is processed through a pipeline: **PDF parsing -> section detection -> hierarchical chunking -> embedding -> Qdrant upsert**.

Chunks are embedded using both dense (BGE) and sparse (SPLADE) models, then stored in Qdrant with metadata as payload:

```python
from qdrant_client.models import PointStruct, SparseVector

client.upsert(
    collection_name="skilllens_resumes",
    points=[
        PointStruct(
            id="uuid-from-chunk-id",
            vector={
                "dense": dense_embedding,           # 768-dim float vector
                "sparse": SparseVector(
                    indices=sparse_indices,          # SPLADE token positions
                    values=sparse_values,            # SPLADE token weights
                ),
            },
            payload={
                "person_name": "john doe",
                "section": "technical skills",
                "content": "Python, PyTorch, LangChain...",
                "resume_id": "john_doe_resume",
            },
        )
    ],
)
```

### 3. Hybrid Search

At query time, the user's query is embedded with the same models and searched against Qdrant using **Reciprocal Rank Fusion (RRF)** to combine dense and sparse results:

```python
from qdrant_client.models import Prefetch, Query, SparseVector

results = client.query_points(
    collection_name="skilllens_resumes",
    prefetch=[
        Prefetch(query=dense_query, using="dense", limit=50),
        Prefetch(
            query=SparseVector(indices=sparse_indices, values=sparse_values),
            using="sparse",
            limit=50,
        ),
    ],
    query=Query(fusion="rrf"),
    limit=50,
    with_payload=True,
)
```

This hybrid approach combines:

- **Dense search** (BGE embeddings): captures semantic meaning ("machine learning" matches "neural networks")
- **Sparse search** (SPLADE): captures exact keyword matches and rare terms
- **RRF fusion**: merges both result sets into a single ranking

### 4. Collection Management

The application uses the Qdrant SDK to list, create, delete, and query multiple collections, allowing users to organize resumes by job posting, department, or batch:

```python
result = client.get_collections()              # List all collections
client.get_collection("collection_name")       # Get collection info
client.delete_collection("collection_name")    # Delete a collection
```

---

## RAG Pipeline Detail

The RAG pipeline in `agent.py` consists of four stages:

### Stage 1: Prompt Enhancement + Count Extraction

The user's natural language query is sent to Gemini, which returns a JSON object containing both the retrieval-optimized query and the desired number of results:

- Input: "find me a good AI engineer"
- Output: `{"query": "AI/ML engineer with deep learning, PyTorch, TensorFlow, NLP...", "count": 1}`

The LLM infers `count` from the user's phrasing: "a candidate" = 1, "top 5" = 5, plural with no number = 3.

### Stage 2: Full Collection Search via Qdrant

The enhanced prompt is embedded and searched against the entire Qdrant collection using hybrid (dense + sparse) search with RRF fusion. The limit is set to the total number of elements in the collection to ensure every chunk in every resume receives a similarity score.

### Stage 3: Resume-Level Ranking (Top-K Mean)

Chunks are grouped by `resume_id`. For each resume, the top 3 highest-scoring chunks are selected and their scores are averaged. This produces a single per-resume score that reflects the resume's best-matching sections without being diluted by irrelevant ones.

```python
# Example scoring:
# Resume A chunks: [0.92, 0.85, 0.78, 0.30, 0.10] -> top 3 mean = 0.85
# Resume B chunks: [0.65, 0.60, 0.55, 0.50, 0.48] -> top 3 mean = 0.60
# Result: Resume A ranks higher
```

### Stage 4: JSON Response

The ranked resumes are returned as a JSON response containing each resume's ID, person name, raw similarity score, number of matched chunks, and filename. The frontend displays these in a ranked table with score bars and clickable links to the original PDF files.

Example response:

```json
{
  "query": "AI/ML engineer with deep learning, PyTorch, LangChain experience",
  "resumes": [
    {
      "resume_id": "john_doe_resume",
      "person_name": "John Doe",
      "score": 0.8523,
      "chunks_matched": 12,
      "filename": "john_doe_resume.pdf"
    },
    {
      "resume_id": "jane_smith_resume",
      "person_name": "Jane Smith",
      "score": 0.7891,
      "chunks_matched": 9,
      "filename": "jane_smith_resume.pdf"
    }
  ]
}
```

---

## Setup and Execution

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- A Google API key for Gemini (get one at https://aistudio.google.com/apikey)

### Step 1: Clone the Repository

```bash
git clone https://github.com/saiteja9078/SkillLens.git
cd SkillLens
```

### Step 2: Start Qdrant via Docker

```bash
docker run -d --name skilllens_qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

Qdrant will start on port 6333. You can verify it's running by visiting the Qdrant dashboard at http://localhost:6333/dashboard.

### Step 3: Set Up the SkillLens Application

Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

- Replace `your_google_api_key_here` with your actual Gemini API key.
- `QDRANT_URL` defaults to `http://localhost:6333` if not set.
- `QDRANT_API_KEY` is optional for local Docker; set it if you configured an API key.

### Step 5: Run the Application

```bash
uvicorn app:app --host 0.0.0.0 --port 5001 --reload
```

Or simply:

```bash
python app.py
```

The server starts at http://localhost:5001. API docs are available at http://localhost:5001/docs.

### Step 6: Use the Application

1. Open http://localhost:5001 in your browser
2. **Upload resumes**: Use the sidebar to drag-and-drop or browse for PDF/DOCX files. Enter a collection name and click "Upload"
3. **Select a collection**: Click a collection in the sidebar or use the dropdown in the header
4. **Delete a collection**: Hover over a collection in the sidebar and click the × icon to delete it
5. **Search**: Type a natural language query in the chat input (e.g., "Find candidates with Python and machine learning experience") and press Enter
6. **Review results**: The interface shows ranked resumes with similarity scores and clickable links to the original PDF files

---

## API Endpoints

| Method | Endpoint              | Description                    |
| ------ | --------------------- | ------------------------------ |
| GET    | `/`                   | Serves the frontend            |
| POST   | `/upload`             | Upload and ingest resume files |
| GET    | `/query`              | RAG query, returns JSON        |
| GET    | `/collections`        | List all Qdrant collections    |
| DELETE | `/collections/{name}` | Delete a Qdrant collection     |
| GET    | `/files/<filename>`   | Serve an uploaded resume file  |

---

## Key Design Decisions

1. **Hybrid search over pure dense search**: Combining dense (semantic) and sparse (keyword) embeddings provides better recall than either approach alone. Dense search captures meaning while sparse search catches exact terms and rare keywords.

2. **RRF (Reciprocal Rank Fusion)**: Qdrant's native RRF fusion combines the dense and sparse result sets into a single ranking without requiring manual weight tuning, producing robust hybrid results out of the box.

3. **Hierarchical chunking**: Resumes are chunked at two levels -- full sections (e.g., all of "Projects") and individual entries (e.g., one specific project). This gives the retrieval system both broad context and fine-grained detail.

4. **Resume-level ranking with top-k mean**: Instead of returning individual chunks, SkillLens scores entire resumes by averaging their top 3 chunk similarity scores. This prevents resumes with more sections from being penalized (mean dilution) and ensures ranking reflects the strongest matching areas.

5. **LLM-controlled result count**: The number of resumes returned is extracted from the user's natural language query by the LLM ("a candidate" = 1, "top 5" = 5), eliminating the need for manual configuration.

6. **Docker-based infrastructure**: Qdrant runs in a Docker container with persistent volume storage, making setup a single `docker run` command.

7. **FastAPI with JSON responses**: Using FastAPI provides automatic OpenAPI documentation, type validation, and clean JSON responses. Returning structured JSON (instead of SSE streaming) makes the API easier to consume by any client -- web, mobile, or programmatic.

---
