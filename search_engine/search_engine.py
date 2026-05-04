"""
Search Engine — Endee Vector Database backend
===============================================
Uses Endee (local instance on port 8080) for vector storage and retrieval.
Embeddings are generated using fastembed (dense + sparse).
"""

from endee import Endee
from fastembed import TextEmbedding, SparseTextEmbedding
import uuid

class SearchEngine:
    def __init__(self, dense_embed=None, sparse_embed=None, collection_name=None):
        self.collection_name = collection_name

        # Connect to local Endee instance
        self.client = Endee()
        self.client.set_base_url("http://127.0.0.1:8080/api/v1")

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

        self._index = None

    def _get_index(self):
        """Get or cache the Endee index handle."""
        if self._index is None and self.collection_name:
            self._index = self.client.get_index(name=self.collection_name)
        return self._index

    def get_total_elements(self):
        """Get total number of vectors in the current collection."""
        try:
            result = self.client.list_indexes()
            for idx in result.get("indexes", []):
                if idx.get("name") == self.collection_name:
                    return idx.get("total_elements", 0)
        except Exception:
            pass
        return 0

    def _create_collection(self, collection_name):
        """Create a new hybrid index in Endee."""
        self.collection_name = collection_name
        try:
            self.client.create_index(
                name=collection_name,
                dimension=768,          # bge-base-en-v1.5 output dim
                sparse_dim=30522,       # SPLADE vocabulary size (BERT wordpiece)
                space_type="cosine",
            )
            self._index = self.client.get_index(name=collection_name)
        except Exception:
            # Index already exists — just get a handle
            self._index = self.client.get_index(name=collection_name)
            return "fail"
        return "success"

    def _push_points(self, docs):
        """Embed documents and upsert into Endee."""
        print("Pushing docs to Endee...")
        index = self._get_index()

        for i, doc in enumerate(docs):
            text = doc["content_with_context"]

            # Generate embeddings
            dense_vector = list(self.dense_embed.embed([text]))[0].tolist()
            sparse_result = list(self.sparse_embed.embed(text))[0]
            sparse_indices = sparse_result.indices.tolist()
            sparse_values = sparse_result.values.tolist()

            # Use chunk_id as the vector id
            point_id = doc["chunk_id"]

            # Build metadata (everything except the embedding-related fields)
            meta = {
                k: v
                for k, v in doc.items()
                if k not in ("vector", "sparse_indices", "sparse_values")
            }

            index.upsert(
                [
                    {
                        "id": point_id,
                        "vector": dense_vector,
                        "sparse_indices": sparse_indices,
                        "sparse_values": sparse_values,
                        "meta": meta,
                    }
                ]
            )

            if (i + 1) % 10 == 0:
                print(f"  Upserted {i + 1}/{len(docs)} chunks")

        print(f"  Done — {len(docs)} chunks upserted.")

    def dense_search(self, query, limit=6):
        """Dense-only similarity search."""
        dense_query = list(self.dense_embed.embed([query]))[0].tolist()
        index = self._get_index()
        return index.query(vector=dense_query, top_k=limit)

    def hybrid_search(self, query, limit=8):
        """Hybrid (dense + sparse) search via Endee."""
        dense_query = list(self.dense_embed.embed([query]))[0].tolist()
        sparse_result = list(self.sparse_embed.embed(query))[0]
        sparse_indices = sparse_result.indices.tolist()
        sparse_values = sparse_result.values.tolist()

        index = self._get_index()
        results = index.query(
            vector=dense_query,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=limit,
        )
        return results