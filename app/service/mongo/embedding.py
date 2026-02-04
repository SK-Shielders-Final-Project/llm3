from __future__ import annotations

from threading import Lock
from typing import Iterable

from app.config.mongoDB import get_mongo_settings

_encoder = None
_encoder_lock = Lock()


def _load_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is not None:
            return _encoder
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - 런타임 환경에서 확인
            raise RuntimeError(
                "sentence_transformers 모듈을 불러올 수 없습니다. "
                "벡터 임베딩 모델을 설치했는지 확인하세요."
            ) from exc
        settings = get_mongo_settings()
        if not settings.embedding_model:
            raise RuntimeError("MONGODB_EMBEDDING_MODEL이 설정되지 않았습니다.")
        _encoder = SentenceTransformer(settings.embedding_model)
        return _encoder


def embed_text(text: str) -> list[float]:
    if not text or not text.strip():
        return []
    encoder = _load_encoder()
    vector = encoder.encode(text).tolist()
    settings = get_mongo_settings()
    if settings.embedding_dim and len(vector) != settings.embedding_dim:
        raise RuntimeError(
            f"임베딩 차원 불일치: expected={settings.embedding_dim} got={len(vector)}"
        )
    return vector


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    encoder = _load_encoder()
    vector_list = encoder.encode(list(texts)).tolist()
    settings = get_mongo_settings()
    if settings.embedding_dim:
        for vector in vector_list:
            if len(vector) != settings.embedding_dim:
                raise RuntimeError(
                    f"임베딩 차원 불일치: expected={settings.embedding_dim} got={len(vector)}"
                )
    return vector_list
