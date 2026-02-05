from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from pymongo import MongoClient, UpdateOne


@dataclass(frozen=True)
class TableConfig:
    table: str
    pk: str
    watermark: str | None
    columns: list[str]
    collection: str


TABLES: dict[str, TableConfig] = {
    "users": TableConfig(
        table="users",
        pk="user_id",
        watermark="updated_at",
        columns=[
            "user_id",
            "username",
            "name",
            "email",
            "phone",
            "total_point",
            "admin_level",
            "created_at",
            "updated_at",
        ],
        collection="users",
    ),
    "rentals": TableConfig(
        table="rentals",
        pk="rental_id",
        watermark="created_at",
        columns=[
            "rental_id",
            "user_id",
            "bike_id",
            "start_time",
            "end_time",
            "total_distance",
            "created_at",
        ],
        collection="rentals",
    ),
    "payments": TableConfig(
        table="payments",
        pk="payment_id",
        watermark="created_at",
        columns=[
            "payment_id",
            "user_id",
            "amount",
            "payment_status",
            "payment_method",
            "transaction_id",
            "created_at",
        ],
        collection="payments",
    ),
    "bikes": TableConfig(
        table="bikes",
        pk="bike_id",
        watermark="updated_at",
        columns=[
            "bike_id",
            "serial_number",
            "model_name",
            "status",
            "latitude",
            "longitude",
            "created_at",
            "updated_at",
        ],
        collection="bikes",
    ),
    "inquiries": TableConfig(
        table="inquiries",
        pk="inquiry_id",
        watermark="updated_at",
        columns=[
            "inquiry_id",
            "user_id",
            "title",
            "content",
            "image_url",
            "file_id",
            "admin_reply",
            "created_at",
            "updated_at",
        ],
        collection="inquiries",
    ),
}


def _get_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} 환경변수가 설정되지 않았습니다.")
    return value


def _get_mongo_client() -> MongoClient:
    uri = _get_env("MONGODB_URI")
    return MongoClient(uri)


def _get_oracle_connection():
    try:
        import oracledb
    except Exception as exc:
        raise RuntimeError(
            "oracledb 모듈이 필요합니다. "
            "pip install oracledb 로 설치하세요."
        ) from exc

    dsn = _get_env("ORACLE_DSN")
    user = _get_env("ORACLE_USER")
    password = _get_env("ORACLE_PASSWORD")
    return oracledb.connect(user=user, password=password, dsn=dsn)


def _iso(dt: Any) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.astimezone(timezone.utc).isoformat()
    return str(dt)


def _snake_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).lower(): value for key, value in row.items()}


def _fetch_rows(
    cursor,
    config: TableConfig,
    watermark_value: datetime | None,
    watermark_pk: int | None,
    since: datetime | None,
) -> Iterable[dict[str, Any]]:
    cols = ", ".join(config.columns)
    base = f"SELECT {cols} FROM {config.table}"
    params: dict[str, Any] = {}
    where_parts: list[str] = []

    if config.watermark:
        wm_col = config.watermark
        if watermark_value is not None:
            where_parts.append(
                f"({wm_col} > :wm) OR ({wm_col} = :wm AND {config.pk} > :pk)"
            )
            params["wm"] = watermark_value
            params["pk"] = watermark_pk or 0
        elif since is not None:
            where_parts.append(f"{wm_col} >= :since")
            params["since"] = since

    if where_parts:
        base = f"{base} WHERE {' AND '.join(where_parts)}"

    if config.watermark:
        base = f"{base} ORDER BY {config.watermark}, {config.pk}"
    else:
        base = f"{base} ORDER BY {config.pk}"

    cursor.execute(base, params)
    col_names = [desc[0] for desc in cursor.description]
    for row in cursor:
        yield {col_names[i]: row[i] for i in range(len(col_names))}


def _load_sync_state(db, table_key: str) -> tuple[datetime | None, int | None]:
    state = db["sync_state"].find_one({"_id": table_key}) or {}
    wm = state.get("last_watermark")
    pk = state.get("last_pk")
    if isinstance(wm, datetime):
        return wm, int(pk) if pk is not None else None
    if isinstance(wm, str) and wm:
        try:
            return datetime.fromisoformat(wm), int(pk) if pk is not None else None
        except ValueError:
            return None, None
    return None, None


def _save_sync_state(db, table_key: str, watermark: datetime | None, pk: int | None) -> None:
    payload = {
        "_id": table_key,
        "last_watermark": watermark or None,
        "last_pk": pk or None,
        "updated_at": datetime.now(tz=timezone.utc),
    }
    db["sync_state"].update_one({"_id": table_key}, {"$set": payload}, upsert=True)


def _upsert_rows(collection, table_key: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ops = []
    now = datetime.now(tz=timezone.utc)
    for row in rows:
        doc = _snake_keys(row)
        pk_val = doc.get(TABLES[table_key].pk)
        doc["oracle_id"] = pk_val
        doc["_source"] = "oracle"
        doc["_table"] = TABLES[table_key].table
        doc["_synced_at"] = now
        ops.append(
            UpdateOne(
                {"oracle_id": pk_val},
                {"$set": doc},
                upsert=True,
            )
        )
    if ops:
        collection.bulk_write(ops, ordered=False)


def run_sync(
    *,
    tables: list[str],
    full: bool,
    batch_size: int,
    since: datetime | None,
    dry_run: bool,
) -> None:
    mongo = _get_mongo_client()
    db_name = _get_env("MONGODB_DB_NAME")
    db = mongo[db_name]

    with _get_oracle_connection() as conn:
        cursor = conn.cursor()
        for table_key in tables:
            config = TABLES[table_key]
            collection = db[config.collection]
            if full:
                collection.delete_many({})
                _save_sync_state(db, table_key, None, None)

            watermark_value, watermark_pk = _load_sync_state(db, table_key)
            rows_buffer: list[dict[str, Any]] = []
            latest_wm = watermark_value
            latest_pk = watermark_pk

            if not config.watermark and not full and since is None:
                print(f"[skip] {table_key}: watermark 없음 (full 또는 since 필요)")
                continue

            for row in _fetch_rows(cursor, config, watermark_value, watermark_pk, since):
                rows_buffer.append(row)
                wm_col = config.watermark
                if wm_col:
                    latest_wm = row.get(wm_col.upper(), row.get(wm_col)) or latest_wm
                    latest_pk = row.get(config.pk.upper(), row.get(config.pk)) or latest_pk
                else:
                    latest_pk = row.get(config.pk.upper(), row.get(config.pk)) or latest_pk

                if len(rows_buffer) >= batch_size:
                    if not dry_run:
                        _upsert_rows(collection, table_key, rows_buffer)
                    rows_buffer = []

            if rows_buffer and not dry_run:
                _upsert_rows(collection, table_key, rows_buffer)

            if not dry_run:
                if isinstance(latest_wm, str):
                    try:
                        latest_wm = datetime.fromisoformat(latest_wm)
                    except ValueError:
                        latest_wm = None
                _save_sync_state(db, table_key, latest_wm, latest_pk)

            print(
                f"[done] {table_key}: full={full} since={_iso(since)} "
                f"last_wm={_iso(latest_wm)} last_pk={latest_pk}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle -> MongoDB 배치 동기화")
    parser.add_argument(
        "--tables",
        type=str,
        default="",
        help="동기화할 테이블 목록 (예: users,rentals). 비우면 전체.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="컬렉션 전체 재적재 (delete_many 후 upsert).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="MongoDB bulk upsert 배치 크기",
    )
    parser.add_argument(
        "--since",
        type=str,
        default="",
        help="증분 기준 시각(ISO-8601). 예: 2026-02-05T00:00:00+09:00",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 반영 없이 실행 로그만 출력",
    )
    return parser.parse_args()


def _parse_since(raw: str) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(raw)


def main() -> None:
    args = _parse_args()
    table_keys = [key.strip() for key in args.tables.split(",") if key.strip()]
    if not table_keys:
        table_keys = list(TABLES.keys())
    for key in table_keys:
        if key not in TABLES:
            raise ValueError(f"알 수 없는 테이블: {key}")

    run_sync(
        tables=table_keys,
        full=bool(args.full),
        batch_size=max(1, int(args.batch_size)),
        since=_parse_since(args.since),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
