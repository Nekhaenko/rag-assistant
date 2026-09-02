from FlagEmbedding import FlagReranker

from src.retrieval import Chunk


class Reranker:
    """Cross-encoder: точная переранжировка кандидатов из гибридного поиска."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        if not chunks:
            return chunks
        scores = self._model.compute_score([[query, c.text] for c in chunks], normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]
