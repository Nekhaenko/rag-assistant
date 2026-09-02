import json

import opentelemetry.trace as otel_trace
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from phoenix.otel import register

from src.config import Settings


def setup_tracing(cfg: Settings) -> otel_trace.Tracer:
    """Phoenix + автотрейсинг всех вызовов OpenAI-клиента (в т.ч. Ollama)."""
    provider = register(
        project_name=cfg.phoenix_project,
        endpoint=f"{cfg.phoenix_endpoint.rstrip('/')}/v1/traces",
    )
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    return otel_trace.get_tracer("rag-assistant")


def docs_json(chunks) -> str:
    """Сериализация чанков в OpenInference-формат для UI Phoenix."""
    return json.dumps(
        [
            {
                "document.id": c.id,
                "document.content": c.text[:1500],
                "document.score": round(c.rerank_score if c.rerank_score is not None else c.score, 4),
            }
            for c in chunks
        ],
        ensure_ascii=False,
    )


__all__ = ["setup_tracing", "docs_json", "SpanAttributes"]
