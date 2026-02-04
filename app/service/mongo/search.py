from __future__ import annotations

from typing import Any

from app.config.mongoDB import get_mongo_collection, get_mongo_settings
from app.service.mongo.embedding import embed_text


def search_knowledge(
    query: str,
    user_id: int | None,
    admin_level: int,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not query or not query.strip():
        return []

    safe_top_k = max(1, min(int(top_k), 20))
    query_vector = embed_text(query)
    if not query_vector:
        return []

    settings = get_mongo_settings()
    collection = get_mongo_collection()

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": settings.vector_index,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(50, safe_top_k * 10),
                "limit": safe_top_k * 3,
                "filter": _build_access_filter(user_id=user_id, admin_level=admin_level),
            }
        },
        {
            "$project": {
                "content": 1,
                "metadata": 1,
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

    return list(collection.aggregate(pipeline))


def _build_access_filter(user_id: int | None, admin_level: int) -> dict[str, Any]:
    if admin_level >= 1:
        return {}

    base_filter: list[dict[str, Any]] = [{"metadata.access_level": "public"}]
    if user_id is not None:
        base_filter.append({"metadata.user_id": user_id})

    access_filter: dict[str, Any] = {"$or": base_filter}

    if user_id is None:
        access_filter = {
            "$and": [
                {"metadata.access_level": "public"},
                {
                    "$or": [
                        {"metadata.requires_auth": False},
                        {"metadata.requires_auth": {"$exists": False}},
                    ]
                },
            ]
        }

    return access_filter
