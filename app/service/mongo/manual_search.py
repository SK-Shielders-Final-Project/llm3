from __future__ import annotations

from typing import Any

from app.config.mongoDB import get_mongo_client, get_mongo_settings
from app.service.mongo.embedding import embed_text
import os


def get_manual_collection_settings() -> dict[str, str]:
    """Manual 컬렉션 설정 반환 (안전수칙, Q&A용)"""
    settings = get_mongo_settings()
    return {
        "collection_name": os.getenv("MANUAL_MONGODB_COLLECTION", "manual"),
        "vector_index": os.getenv("MANUAL_MONGODB_VECTOR_INDEX", "manual_index"),
        "db_name": settings.db_name,
    }


def search_manual_knowledge(
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Manual 컬렉션에서 안전수칙/Q&A 문서를 Vector Search로 검색.
    manual/main.py로 저장된 문서 스키마에 맞춤:
    - type: "qna" | "safety_rule"
    - title: 제목/질문
    - content: 답변/내용
    - embedding: 벡터
    """
    if not query or not query.strip():
        return []

    safe_top_k = max(1, min(int(top_k), 20))
    query_vector = embed_text(query)
    if not query_vector:
        return []

    manual_settings = get_manual_collection_settings()
    client = get_mongo_client()
    collection = client[manual_settings["db_name"]][manual_settings["collection_name"]]

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": manual_settings["vector_index"],
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(50, safe_top_k * 10),
                "limit": safe_top_k * 2,
            }
        },
        {
            "$project": {
                "type": 1,
                "title": 1,
                "content": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
        {
            "$addFields": {
                "score": {"$ifNull": ["$score", 0.0]},
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": safe_top_k},
    ]

    try:
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        # Manual 컬렉션 검색 실패 시 빈 결과 반환 (기존 로직 방해 안 함)
        import logging
        logging.getLogger("manual_search").warning(
            "Manual 컬렉션 검색 실패: %s", str(e)
        )
        return []


def format_manual_results_as_context(results: list[dict[str, Any]]) -> str:
    """Manual 검색 결과를 LLM 컨텍스트 문자열로 변환"""
    if not results:
        return ""
    
    parts = ["=== 관련 문서 (FAQ/안전수칙) ==="]
    for doc in results:
        doc_type = doc.get("type", "unknown")
        title = doc.get("title", "")
        content = doc.get("content", "")
        score = doc.get("score", 0.0)
        
        if doc_type == "qna":
            parts.append(f"Q: {title}")
            parts.append(f"A: {content}")
        elif doc_type == "safety_rule":
            parts.append(f"[{title}] {content}")
        else:
            parts.append(f"{title}: {content}")
    
    return "\n".join(parts)
