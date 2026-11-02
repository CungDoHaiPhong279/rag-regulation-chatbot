# Project Rules: RAG Regulation Chatbot

## Architecture Overview
- **Ingestion** (src/ingestion/): Parse regulation PDFs, hierarchical/semantic chunking (chunker.py), and indexing into Qdrant (indexer.py).
- **Retrieval** (src/retrieval/): Hybrid search combining dense + sparse/BM25 vectors (hybrid_search.py) and cross-encoder reranking (eranker.py).
- **API** (src/api/main.py): FastAPI backend serving RAG query endpoints with citations.
- **UI** (src/ui/app.py): Chat interface displaying answers and exact regulation references.
- **Evaluation** (eval/evaluate_recall.py): Recall@K and MRR evaluation scripts.

## Coding Standards
- Python 3.10+, use type hints and docstrings for all functions/classes.
- Never hardcode API keys or secrets; load from .env via python-dotenv.
- Always preserve article/clause metadata (Điều, Khoản, Điểm, Trang) during chunking and retrieval.
