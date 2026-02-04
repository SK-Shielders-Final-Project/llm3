from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config.mongoDB import get_mongo_collection
from app.service.mongo.embedding import embed_text


def store_user_message(
    *,
    user_id: int,
    content: str,
    role: str = "user",
    doc_type: str = "conversation",
    category: str = "chat_history",
    access_level: str = "user",
    requires_auth: bool = True,
    importance: int = 3,
    intent_tags: list[str] | None = None,
) -> str | None:
    if not content or not content.strip():
        return None

    collection = get_mongo_collection()
    embedding = embed_text(content)
    now = datetime.now(tz=timezone.utc)
    payload: dict[str, Any] = {
        "content": content,
        "embedding": embedding,
        "metadata": {
            "doc_type": doc_type,
            "category": category,
            "requires_mysql": False,
            "mysql_tables": [],
            "access_level": access_level,
            "requires_auth": requires_auth,
            "importance": max(1, min(int(importance), 10)),
            "freshness_score": 1.0,
            "intent_tags": intent_tags or ["chat_history"],
            "role": role,
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
        },
    }
    result = collection.insert_one(payload)
    return str(result.inserted_id) if result.inserted_id else None
