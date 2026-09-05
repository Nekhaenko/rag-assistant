import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from openai import OpenAI
from qdrant_client import QdrantClient, models

from src.config import Settings
from src.embedders import M3Embedder
from src.llm import chat

_NOTES_PROMPT = """Ты — модуль долговременной памяти ассистента. Из диалога извлеки устойчивые
факты, которые стоит запомнить: предпочтения и контекст пользователя, принятые решения, важные
ограничения. Общие вопросы без ценности не извлекай. Если запоминать нечего — верни пустой список.
Верни строго JSON: {"notes": ["...", "..."]}"""


class ChatMemory:
    """Краткосрочная память: окно последних ходов на сессию."""

    def __init__(self, window: int = 6):
        self.window = window
        self._sessions: dict[str, list[dict]] = defaultdict(list)

    def add(self, session_id: str, role: str, content: str) -> None:
        self._sessions[session_id].append({"role": role, "content": content})

    def history(self, session_id: str) -> list[dict]:
        return self._sessions[session_id][-self.window * 2:]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class LongTermMemory:
    """Семантическая память: заметки в отдельной коллекции Qdrant.
    После каждого хода LLM извлекает факты, они векторизуются и складываются;
    перед ответом достаём релевантные воспоминания."""

    def __init__(self, client: QdrantClient, embedder: M3Embedder,
                 cfg: Settings, llm: OpenAI):
        self.client, self.embedder, self.cfg, self.llm = client, embedder, cfg, llm
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.cfg.memory_collection):
            self.client.create_collection(
                self.cfg.memory_collection,
                vectors_config={
                    "dense": models.VectorParams(size=self.cfg.dense_dim,
                                                 distance=models.Distance.COSINE)
                },
            )

    def remember(self, session_id: str, question: str, answer: str) -> list[str]:
        try:
            raw = chat(self.llm, self.cfg.chat_model,
                       [{"role": "system", "content": _NOTES_PROMPT},
                        {"role": "user", "content": f"Пользователь: {question}\nАссистент: {answer}"}],
                       temperature=0, json_mode=True)
            notes = [n.strip() for n in json.loads(raw).get("notes", []) if n.strip()][:5]
        except Exception:
            return []

        embedded = self.embedder.embed_documents(notes)
        points = []
        for note, e in zip(notes, embedded):
            hits = self.client.query_points(self.cfg.memory_collection,
                                            query=e.dense, using="dense", limit=1).points
            if hits and hits[0].score > 0.97:  # дедуп близких заметок
                continue
            points.append(models.PointStruct(
                id=str(uuid.uuid4()), vector={"dense": e.dense},
                payload={"text": note, "session_id": session_id,
                         "created_at": datetime.now(timezone.utc).isoformat()},
            ))
        if points:
            self.client.upsert(self.cfg.memory_collection, points=points)
        return notes

    def recall(self, query: str, k: int = 3) -> list[str]:
        if not self.client.collection_exists(self.cfg.memory_collection):
            return []
        e = self.embedder.embed_query(query)
        hits = self.client.query_points(self.cfg.memory_collection,
                                        query=e.dense, using="dense", limit=k).points
        return [h.payload["text"] for h in hits if h.score > 0.3]