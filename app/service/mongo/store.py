from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from app.config.mongoDB import get_mongo_collection
from app.service.mongo.embedding import embed_text

logger = logging.getLogger(__name__)


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
    try:
        embedding = embed_text(content)
    except Exception as exc:  # pragma: no cover - 런타임 환경에서 확인
        logger.exception("MongoDB 임베딩 생성 실패, 빈 임베딩으로 저장합니다.")
        embedding = []
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
    try:
        result = collection.insert_one(payload)
    except Exception:  # pragma: no cover - 런타임 환경에서 확인
        logger.exception("MongoDB 저장 실패")
        raise
    return str(result.inserted_id) if result.inserted_id else None
