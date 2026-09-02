from openai import OpenAI

from src.config import Settings


def make_client(cfg: Settings) -> OpenAI:
    # Ollama отдаёт OpenAI-совместимый API; api_key — любая непустая строка
    return OpenAI(base_url=cfg.ollama_base_url, api_key="ollama")


def chat(client: OpenAI, model: str, messages: list[dict],
         temperature: float = 0.2, json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, **kwargs
    )
    return resp.choices[0].message.content or ""
