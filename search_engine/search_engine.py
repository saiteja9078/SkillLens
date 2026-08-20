"""
Search Engine — Qdrant Vector Database backend
=================================================
Uses Qdrant (via Docker) for vector storage and retrieval.
Embeddings are generated using fastembed (dense + sparse).
"""

import os
import uuid

from dotenv import load_dotenv
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None


def _md5_to_uuid(md5_hex: str) -> str:
    """Convert a 32-char MD5 hex string into a valid UUID string."""
    return str(uuid.UUID(md5_hex))


class SearchEngine:
    def __init__(self, dense_embed=None, sparse_embed=None, collection_name=None):
        self.collection_name = collection_name

        # Connect to Qdrant
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Embedding models
        self.dense_embed = (
            TextEmbedding("BAAI/bge-base-en-v1.5")
            if dense_embed is None
            else dense_embed
        )
        self.sparse_embed = (
            SparseTextEmbedding("prithvida/Splade_PP_en_v1")
            if sparse_embed is None
            else sparse_embed
        )

    def get_total_elements(self):
        """Get total number of vectors in the current collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def _create_collection(self, collection_name):
        """Create a new hybrid collection in Qdrant."""
        self.collection_name = collection_name
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=768,  # bge-base-en-v1.5 output dim
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False),
                    ),
                },
            )
        except Exception:
            # Collection already exists — reuse it
            return "fail"
        return "success"

    def _push_points(self, docs):
        """Embed documents and upsert into Qdrant."""
        print("Pushing docs to Qdrant...")

        batch_size = 64
        points = []

        for i, doc in enumerate(docs):
            text = doc["content_with_context"]

            # Generate embeddings
            dense_vector = list(self.dense_embed.embed([text]))[0].tolist()
            sparse_result = list(self.sparse_embed.embed(text))[0]
            sparse_indices = sparse_result.indices.tolist()
            sparse_values = sparse_result.values.tolist()

            # Convert MD5 hex chunk_id to UUID
            point_id = _md5_to_uuid(doc["chunk_id"])

            # Build payload (everything except embedding-related fields)
            payload = {
                k: v
                for k, v in doc.items()
                if k not in ("vector", "sparse_indices", "sparse_values")
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vector,
                        "sparse": SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload=payload,
                )
            )

            # Batch upsert
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                print(f"  Upserted {i + 1}/{len(docs)} chunks")
                points = []

        # Flush remaining
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        print(f"  Done — {len(docs)} chunks upserted.")

    def dense_search(self, query, limit=6):
        """Dense-only similarity search."""
        dense_query = list(self.dense_embed.embed([query]))[0].tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_query,
            using="dense",
            limit=limit,
            with_payload=True,
        ).points

        return self._format_results(results)

    def hybrid_search(self, query, limit=8):
        """Hybrid (dense + sparse) search via Qdrant using RRF fusion."""
        dense_query = list(self.dense_embed.embed([query]))[0].tolist()
        sparse_result = list(self.sparse_embed.embed(query))[0]
        sparse_indices = sparse_result.indices.tolist()
        sparse_values = sparse_result.values.tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=limit,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=limit,
                ),
            ],
            query=Fusion.RRF,
            limit=limit,
            with_payload=True,
        ).points

        return self._format_results(results)

    @staticmethod
    def _format_results(points):
        """
        Convert Qdrant query results into the same format
        the rest of the codebase expects:
          [{"meta": {...}, "similarity": float}, ...]
        """
        formatted = []
        for pt in points:
            formatted.append({
                "meta": pt.payload or {},
                "similarity": pt.score if pt.score is not None else 0.0,
            })
        return formatted