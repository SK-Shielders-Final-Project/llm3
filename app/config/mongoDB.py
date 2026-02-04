from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock
from typing import Any

from pymongo import MongoClient

_client: MongoClient | None = None
_client_lock = Lock()


@dataclass(frozen=True)
class MongoSettings:
    uri: str
    db_name: str
    collection_name: str
    vector_index: str
    embedding_model: str
    embedding_dim: int
    connect_timeout_ms: int
    server_selection_timeout_ms: int


def get_mongo_settings() -> MongoSettings:
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError("MONGODB_URI가 설정되지 않았습니다.")

    db_name = os.getenv("MONGODB_DB_NAME", "").strip()
    collection_name = os.getenv("MONGODB_COLLECTION", "").strip()
    vector_index = os.getenv("MONGODB_VECTOR_INDEX", "").strip()
    embedding_model = os.getenv("MONGODB_EMBEDDING_MODEL", "").strip()
    embedding_dim_raw = os.getenv("MONGODB_EMBEDDING_DIM", "0").strip()
    connect_timeout_raw = os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "2000").strip()
    server_selection_timeout_raw = os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "2000").strip()

    embedding_dim = int(embedding_dim_raw) if embedding_dim_raw else 0
    connect_timeout_ms = int(connect_timeout_raw) if connect_timeout_raw else 2000
    server_selection_timeout_ms = (
        int(server_selection_timeout_raw) if server_selection_timeout_raw else 2000
    )

    return MongoSettings(
        uri=uri,
        db_name=db_name,
        collection_name=collection_name,
        vector_index=vector_index,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        connect_timeout_ms=connect_timeout_ms,
        server_selection_timeout_ms=server_selection_timeout_ms,
    )


def get_mongo_client() -> MongoClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                settings = get_mongo_settings()
                _client = MongoClient(
                    settings.uri,
                    connectTimeoutMS=settings.connect_timeout_ms,
                    serverSelectionTimeoutMS=settings.server_selection_timeout_ms,
                )
    return _client


def get_mongo_collection() -> Any:
    settings = get_mongo_settings()
    client = get_mongo_client()
    return client[settings.db_name][settings.collection_name]
