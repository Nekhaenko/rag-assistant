from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from src.config import Settings
from src.embedders import Embedded, M3Embedder


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    score: float
    rerank_score: float | None = None


class HybridRetriever:
    """Гибридный поиск: dense + sparse параллельно, фьюжн RRF на стороне Qdrant."""

    def __init__(self, client: QdrantClient, embedder: M3Embedder, cfg: Settings):
        self.client, self.embedder, self.cfg = client, embedder, cfg

    def search(self, query: str, top_k: int, source: str | None = None) -> list[Chunk]:
        emb: Embedded = self.embedder.embed_query(query)
        flt = None
        if source:
            flt = models.Filter(must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=source))
            ])
        prefetch = [
            models.Prefetch(query=emb.dense, using="dense",
                            limit=self.cfg.top_k_dense, filter=flt),
            models.Prefetch(query=emb.sparse, using="sparse",
                            limit=self.cfg.top_k_sparse, filter=flt),
        ]
        res = self.client.query_points(
            self.cfg.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [
            Chunk(
                id=str(p.id),
                text=p.payload["text"],
                source=p.payload["source"],
                chunk_index=p.payload.get("chunk_index", 0),
                score=p.score,
            )
            for p in res.points
        ]
