import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.config import Settings
from src.embedders import M3Embedder

_SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Абзац-ориентированный чанкер с перекрытием по хвосту."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}".strip()
            continue
        if buf:
            chunks.append(buf)
        if len(para) > size:  # разбиваем по предложениям
            cur = ""
            for s in _SENT_RE.split(para):
                if len(cur) + len(s) + 1 <= size:
                    cur = f"{cur} {s}".strip()
                else:
                    if cur:
                        chunks.append(cur)
                    cur = s
            buf = cur
        else:
            buf = para
    if buf:
        chunks.append(buf)
    if overlap and len(chunks) > 1:
        chunks = [chunks[0]] + [
            f"{prev[-overlap:]}\n\n{c}" for prev, c in zip(chunks, chunks[1:])
        ]
    return chunks


def _ensure_collection(client: QdrantClient, cfg: Settings) -> None:
    if client.collection_exists(cfg.collection):
        return
    client.create_collection(
        cfg.collection,
        vectors_config={
            "dense": models.VectorParams(size=cfg.dense_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
    )


def ingest_dir(cfg: Settings, folder: str = "data/docs") -> None:
    client = QdrantClient(url=cfg.qdrant_url)
    embedder = M3Embedder(cfg.dense_model)
    _ensure_collection(client, cfg)

    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"}]
    if not files:
        raise FileNotFoundError(f"Нет документов в {folder}")

    total = 0
    for f in files:
        chunks = chunk_text(read_text(f), cfg.chunk_size, cfg.chunk_overlap)
        if not chunks:
            continue
        embedded = embedder.embed_documents(chunks)

        # идемпотентность: сносим старые чанки этого файла
        client.delete(
            cfg.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(key="path", match=models.MatchValue(value=str(f)))
                ])
            ),
        )
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{f}#{i}")),
                vector={"dense": e.dense, "sparse": e.sparse},
                payload={"text": c, "source": f.name, "path": str(f), "chunk_index": i},
            )
            for i, (c, e) in enumerate(zip(chunks, embedded))
        ]
        client.upsert(cfg.collection, points=points)
        total += len(chunks)
        print(f"  {f.name}: {len(chunks)} чанков")

    print(f"Готово: {total} чанков из {len(files)} файлов")
