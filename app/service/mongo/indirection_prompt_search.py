from __future__ import annotations

import logging
import os
from typing import Any

from app.config.mongoDB import get_mongo_client, get_mongo_settings
from app.service.mongo.embedding import embed_text

logger = logging.getLogger("indirection_prompt")


def _is_enabled() -> bool:
    """INDIRECTION_PROMPT_ENABLED 환경변수로 기능 ON/OFF 제어."""
    return os.getenv("INDIRECTION_PROMPT_ENABLED", "false").strip().lower() in {
        "true", "1", "yes",
    }


def _get_settings() -> dict[str, Any]:
    """Indirection Prompt 컬렉션 관련 설정 반환."""
    settings = get_mongo_settings()
    return {
        "collection_name": os.getenv("INDIRECTION_PROMPT_COLLECTION", "indirection_prompt"),
        "vector_index": os.getenv("INDIRECTION_PROMPT_VECTOR_INDEX", "indirection_prompt_index"),
        "threshold": float(os.getenv("INDIRECTION_PROMPT_THRESHOLD", "0.80")),
        "db_name": settings.db_name,
    }


def search_indirection_prompt(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    indirection_prompt 컬렉션에서 Vector Search를 수행한다.
    Scanner에서 우회 프롬프트를 Human-in-the-loop 검증 후 임베딩한 문서를 검색한다.
    """
    if not query or not query.strip():
        return []

    query_vector = embed_text(query)
    if not query_vector:
        return []

    ip_settings = _get_settings()
    client = get_mongo_client()
    collection = client[ip_settings["db_name"]][ip_settings["collection_name"]]

    safe_top_k = max(1, min(int(top_k), 10))

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": ip_settings["vector_index"],
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(50, safe_top_k * 10),
                "limit": safe_top_k,
            }
        },
        {
            "$project": {
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
        return list(collection.aggregate(pipeline))
    except Exception as e:
        logger.warning("indirection_prompt 컬렉션 검색 실패: %s", str(e))
        return []


def check_indirection_prompt(query: str) -> dict[str, Any]:
    """
    사용자 프롬프트가 우회 프롬프트와 유사한지 검사한다.

    Returns:
        {
            "blocked": bool,       # 차단 여부
            "score": float,        # 최고 유사도 점수
            "matched_content": str, # 매칭된 우회 프롬프트 내용 (차단 시)
            "threshold": float,    # 적용된 임계값
        }
    """
    if not _is_enabled():
        return {"blocked": False, "score": 0.0, "matched_content": "", "threshold": 0.0}

    ip_settings = _get_settings()
    threshold = ip_settings["threshold"]

    results = search_indirection_prompt(query, top_k=1)
    if not results:
        return {"blocked": False, "score": 0.0, "matched_content": "", "threshold": threshold}

    top_result = results[0]
    score = float(top_result.get("score", 0.0))
    content = str(top_result.get("content", ""))

    blocked = score >= threshold

    if blocked:
        logger.info(
            "우회 프롬프트 탐지 - 차단 score=%.4f threshold=%.2f matched=%s",
            score, threshold, content[:100],
        )
    else:
        logger.debug(
            "우회 프롬프트 미탐지 score=%.4f threshold=%.2f",
            score, threshold,
        )

    return {
        "blocked": blocked,
        "score": score,
        "matched_content": content if blocked else "",
        "threshold": threshold,
    }
