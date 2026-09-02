from dataclasses import dataclass

import torch
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import models


@dataclass
class Embedded:
    dense: list[float]
    sparse: models.SparseVector


class M3Embedder:
    """BGE-M3: одна модель даёт и dense (1024), и sparse (lexical weights) вектора.
    Мультиязычная, хорошо работает с русским."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self._model = BGEM3FlagModel(model_name, use_fp16=torch.cuda.is_available())

    @staticmethod
    def _to_sparse(lexical: dict) -> models.SparseVector:
        if not lexical:
            return models.SparseVector(indices=[], values=[])
        return models.SparseVector(
            indices=[int(tok) for tok in lexical.keys()],
            values=[float(w) for w in lexical.values()],
        )

    def embed(self, texts: list[str], is_query: bool = False) -> list[Embedded]:
        out = self._model.encode(
            texts,
            batch_size=8,
            max_length=1024 if is_query else 2048,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [
            Embedded(dense=out["dense_vecs"][i].tolist(),
                     sparse=self._to_sparse(out["lexical_weights"][i]))
            for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> Embedded:
        return self.embed([text], is_query=True)[0]

    def embed_documents(self, texts: list[str]) -> list[Embedded]:
        return self.embed(texts)
