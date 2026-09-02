# rag-assistant

rag-assistant/
├── docker-compose.yml
├── requirements.txt
├── data/docs/          # ваша база знаний (.md, .txt, .pdf)
├── eval/golden.jsonl   # тестовые вопросы + эталонные ответы
└── src/
    ├── __init__.py
    ├── config.py
    ├── tracing.py
    ├── llm.py
    ├── embedders.py
    ├── ingest.py
    ├── retrieval.py
    ├── rerank.py
    ├── memory.py
    ├── pipeline.py
    ├── evaluate.py
    └── main.py
