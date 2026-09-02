from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama (OpenAI-compatible API)
    ollama_base_url: str = "http://localhost:11434/v1"
    chat_model: str = "qwen2.5:7b-instruct"
    judge_model: str = "qwen2.5:14b-instruct"   # для оценки качества

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    collection: str = "knowledge"
    memory_collection: str = "memories"

    # Модели эмбеддингов / реранкинга
    dense_model: str = "BAAI/bge-m3"            # dense + sparse, мультиязычная
    dense_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = True

    # Ретривал
    top_k_dense: int = 30
    top_k_sparse: int = 30
    retrieve_k: int = 20          # после RRF-фьюжна
    top_k_after_rerank: int = 5

    # Чанкинг
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Память
    history_window: int = 6       # последних ходов диалога
    memory_top_k: int = 3

    # Phoenix
    phoenix_endpoint: str = "http://localhost:6006"
    phoenix_project: str = "rag-assistant"


settings = Settings()
