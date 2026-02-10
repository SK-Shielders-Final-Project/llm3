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
    qna_id: str | None = None,
    session_id: str | None = None,
    turn_index: int | None = None,
    feedback_score: float | None = None,
    topic_tags: list[str] | None = None,
    entities: list[str] | None = None,
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
    normalized_feedback = _normalize_feedback_score(feedback_score)
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
            "qna_id": qna_id,
            "session_id": session_id,
            "turn_index": int(turn_index) if turn_index is not None else None,
            "feedback_score": normalized_feedback,
            "topic_tags": topic_tags or [],
            "entities": entities or [],
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


def _normalize_feedback_score(raw: float | None) -> float:
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(value, 1.0))
