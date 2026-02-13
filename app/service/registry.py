from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable

from app.config.db import fetch_all, fetch_one
from app.service.mongo.search import search_knowledge as mongo_search_knowledge

class FunctionRegistry:
    """
    FastAPI는 함수 레지스트리 역할만 수행한다.
    실제 실행 주체는 LLM이며, 이 레지스트리는 호출 가능한 함수 목록을 제공한다.
    """
    
    def __init__(self) -> None:
        self._functions: dict[str, Callable[..., Any]] = {
            "get_nearby_stations": get_nearby_stations,
            "get_user_profile": get_user_profile,
            "get_payments": get_payments,
            "get_rentals": get_rentals,
            "get_pricing_summary": get_pricing_summary,
            "get_usage_summary": get_usage_summary,
            "get_available_bikes": get_available_bikes,
            "get_notices": get_notices,
            "get_inquiries": get_inquiries,
            "get_total_payments": get_total_payments,
            "get_total_usage": get_total_usage,
            "search_knowledge": search_knowledge,
            "execute_sql_readonly": execute_sql_readonly,
        }

    def list_functions(self) -> list[str]:
        return sorted(self._functions.keys())

    def execute(self, name: str, **kwargs: Any) -> Any:
        if name not in self._functions:
            raise ValueError(f"Unknown function: {name}")
        func = self._functions[name]
        sig = inspect.signature(func)
        filtered = {key: value for key, value in kwargs.items() if key in sig.parameters}
        if "user_id" in filtered:
            filtered["user_id"] = _coerce_user_id(filtered["user_id"])
        return func(**filtered)




def get_nearby_stations(lat: float, lon: float) -> list[dict[str, Any]]:
    """
    지정된 좌표 근처의 자전거 위치 정보를 반환합니다.
    (스테이션 테이블이 없으므로 bikes 테이블을 기준으로 근사)
    """
    radius_km = 1.0
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / 111.0
    query = (
        "SELECT bike_id, serial_number, model_name, status, latitude, longitude "
        "FROM bikes "
        "WHERE status = 'AVAILABLE' "
        "AND latitude BETWEEN :min_lat AND :max_lat "
        "AND longitude BETWEEN :min_lon AND :max_lon "
        "FETCH FIRST 20 ROWS ONLY"
    )
    return fetch_all(
        query,
        {
            "min_lat": float(lat) - lat_delta,
            "max_lat": float(lat) + lat_delta,
            "min_lon": float(lon) - lon_delta,
            "max_lon": float(lon) + lon_delta,
        },
    )


def get_user_profile(user_id: int, admin_level: int = 0) -> dict[str, Any]:
    """
    users 테이블 기반 사용자 프로필을 조회합니다.
    민감 정보(password, card_number, pass)는 반환하지 않습니다.
    admin_level에 따라 노출 컬럼이 달라집니다:
      - admin_level=0 (일반): user_id, admin_level 숨김
      - admin_level>=1 (관리자): 전체 컬럼 (password/card_number/pass 제외)
    """
    # 관리자: 전체 컬럼 반환 (password/card_number/pass 제외)
    if admin_level >= 1:
        query = (
            "SELECT user_id, username, name, email, phone, total_point, admin_level, created_at, updated_at "
            "FROM users WHERE user_id = :user_id"
        )
        return fetch_one(query, {"user_id": user_id}) or {}

    # 일반 사용자: user_id, admin_level 제외 (취약 모드에서도 동일)
    query = (
        "SELECT username, name, email, phone, total_point, created_at, updated_at "
        "FROM users WHERE user_id = :user_id"
    )
    return fetch_one(query, {"user_id": user_id}) or {}


def get_payments(user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    """payments 테이블 기반 결제 내역을 반환합니다."""
    safe_limit = max(1, min(int(limit), 50))
    query = (
        "SELECT payment_id, user_id, amount, remain_amount, payment_status, payment_method, "
        "order_id, payment_key, created_at "
        "FROM payments WHERE user_id = :user_id "
        "ORDER BY created_at DESC "
        f"FETCH FIRST {safe_limit} ROWS ONLY"
    )
    return fetch_all(query, {"user_id": user_id})


def get_rentals(user_id: int, days: int = 7) -> list[dict[str, Any]]:
    """rentals 테이블 기반 최근 N일 대여 내역을 반환합니다."""
    safe_days = max(1, min(int(days), 30))
    query = (
        "SELECT rental_id, user_id, bike_id, start_time, end_time, total_distance "
        "FROM rentals "
        "WHERE user_id = :user_id AND start_time >= (SYSDATE - :days) "
        "ORDER BY start_time DESC"
    )
    return fetch_all(query, {"user_id": user_id, "days": safe_days})


def get_pricing_summary(user_id: int) -> dict[str, Any]:
    """요금 요약 정보를 반환합니다."""
    query = (
        "SELECT NVL(SUM(amount), 0) AS total_paid, MAX(created_at) AS last_payment_at "
        "FROM payments WHERE user_id = :user_id"
    )
    result = fetch_one(query, {"user_id": user_id}) or {"total_paid": 0, "last_payment_at": None}
    return {"user_id": user_id, "total_paid": result["total_paid"], "currency": "KRW", "last_payment_at": result["last_payment_at"]}


def get_usage_summary(user_id: int) -> dict[str, Any]:
    """이용 요약 정보를 반환합니다."""
    query = (
        "SELECT COUNT(*) AS rental_count, NVL(SUM(total_distance), 0) AS total_distance "
        "FROM rentals WHERE user_id = :user_id"
    )
    result = fetch_one(query, {"user_id": user_id}) or {"rental_count": 0, "total_distance": 0}
    return {"user_id": user_id, **result}


def get_available_bikes(
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 1.0,
) -> list[dict[str, Any]]:
    """대여 가능한 자전거 목록을 반환합니다."""
    safe_radius = max(0.1, min(float(radius_km), 10.0))
    query = (
        "SELECT bike_id, serial_number, model_name, status, latitude, longitude "
        "FROM bikes WHERE status = 'AVAILABLE'"
    )
    if lat is None or lon is None:
        return fetch_all(f"{query} FETCH FIRST 20 ROWS ONLY")

    lat_delta = safe_radius / 111.0
    lon_delta = safe_radius / 111.0
    query = (
        f"{query} AND latitude BETWEEN :min_lat AND :max_lat "
        "AND longitude BETWEEN :min_lon AND :max_lon "
        "FETCH FIRST 50 ROWS ONLY"
    )
    return fetch_all(
        query,
        {
            "min_lat": float(lat) - lat_delta,
            "max_lat": float(lat) + lat_delta,
            "min_lon": float(lon) - lon_delta,
            "max_lon": float(lon) + lon_delta,
        },
    )


def get_notices(limit: int = 5) -> list[dict[str, Any]]:
    """현재 스키마에는 notices 테이블이 없어 빈 목록을 반환합니다."""
    _ = limit
    return []


def get_inquiries(user_id: int) -> list[dict[str, Any]]:
    """inquiries 테이블 기반 문의 내역을 반환합니다."""
    query = (
        "SELECT inquiry_id, user_id, title, content, image_url, file_id, admin_reply, created_at, updated_at "
        "FROM inquiries WHERE user_id = :user_id ORDER BY created_at DESC"
    )
    return fetch_all(query, {"user_id": user_id})


def get_total_payments(user_id: int) -> dict[str, Any]:
    """사용자 전체 결제 합계를 반환합니다."""
    query = "SELECT NVL(SUM(amount), 0) AS total_payments FROM payments WHERE user_id = :user_id"
    result = fetch_one(query, {"user_id": user_id}) or {"total_payments": 0}
    return {"user_id": user_id, "total_payments": result["total_payments"]}


def get_total_usage(user_id: int) -> dict[str, Any]:
    """사용자 전체 이용 합계를 반환합니다."""
    query = (
        "SELECT COUNT(*) AS total_rentals, NVL(SUM(total_distance), 0) AS total_distance "
        "FROM rentals WHERE user_id = :user_id"
    )
    result = fetch_one(query, {"user_id": user_id}) or {"total_rentals": 0, "total_distance": 0}
    return {"user_id": user_id, **result}


def search_knowledge(
    query: str,
    user_id: int,
    admin_level: int = 0,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    MongoDB Vector Search 기반 지식 검색.
    access_level(public/user/admin)을 필터링한 후 결과를 반환합니다.
    """
    profile = get_user_profile(user_id) or {}
    resolved_admin = int(profile.get("admin_level", 0) or 0)
    return mongo_search_knowledge(
        query=query,
        user_id=_coerce_user_id(user_id),
        admin_level=resolved_admin,
        top_k=top_k,
    )


def execute_sql_readonly(query: str, user_id: int) -> list[dict[str, Any]]:
    """
    SELECT 전용 SQL 실행. 민감 컬럼/쓰기 쿼리 차단.
    개인 테이블은 user_id 조건이 반드시 포함되어야 한다.
    """
    logger = logging.getLogger("registry")
    vulnerable_excessive = os.getenv("VULNERABLE_EXCESSIVE_AGENCY", "false").strip().lower() in {"true", "1", "yes"}

    if vulnerable_excessive:
        # ── Excessive Agency 취약점 ──
        # 서버사이드 가드레일(user_id 강제, 민감컬럼 차단)을 제거하고
        # LLM 판단에만 의존한다. LLM이 생성한 SQL을 최소 검증만 수행.
        safe_query = _sanitize_sql_query_permissive(query)
        logger.warning(
            "[VULN] Excessive Agency: 서버사이드 SQL 제한 해제 query=%s",
            safe_query[:200],
        )
    else:
        safe_query = _sanitize_sql_query(query)

    params: dict[str, Any] = {}
    if ":user_id" in safe_query:
        params["user_id"] = _coerce_user_id(user_id)
    return fetch_all(safe_query, params)


def _coerce_user_id(user_id: Any) -> int:
    try:
        value = int(user_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("user_id는 정수여야 합니다.") from exc
    if value <= 0:
        raise ValueError("user_id는 1 이상의 값이어야 합니다.")
    return value


def _sanitize_sql_query_permissive(query: str) -> str:
    """
    Excessive Agency 취약 모드 전용 SQL 검증.
    SELECT 여부와 DDL/DML 차단만 수행하며,
    user_id 강제 바인딩, 민감 컬럼 차단은 수행하지 않는다.
    → LLM이 자유롭게 cross-user 쿼리를 생성·실행할 수 있다.
    """
    if not query or not query.strip():
        raise ValueError("query가 비어 있습니다.")
    raw = query.strip().rstrip(";")
    lowered = raw.lower()
    forbidden = ("insert", "update", "delete", "drop", "alter", "merge", "commit", "rollback")
    if not lowered.startswith("select"):
        raise ValueError("SELECT 쿼리만 허용됩니다.")
    if any(keyword in lowered for keyword in forbidden):
        raise ValueError("쓰기/DDL 쿼리는 허용되지 않습니다.")
    # ── 취약점 핵심: user_id 강제 조건 없음, 민감 컬럼 차단 없음 ──
    # LLM이 "SELECT * FROM users WHERE admin_level >= 1" 같은
    # cross-user 쿼리를 그대로 실행할 수 있다.
    if "fetch first" not in lowered and "limit" not in lowered:
        raw = f"{raw} FETCH FIRST 50 ROWS ONLY"
    return raw


def _sanitize_sql_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("query가 비어 있습니다.")
    raw = query.strip().rstrip(";")
    lowered = raw.lower()
    forbidden = ("insert", "update", "delete", "drop", "alter", "merge", "commit", "rollback")
    if not lowered.startswith("select"):
        raise ValueError("SELECT 쿼리만 허용됩니다.")
    if any(keyword in lowered for keyword in forbidden):
        raise ValueError("쓰기/DDL 쿼리는 허용되지 않습니다.")

    sensitive_cols = ("password", "card_number", "pass")
    if any(col in lowered for col in sensitive_cols):
        raise ValueError("민감 컬럼 조회는 허용되지 않습니다.")

    personal_tables = ("users", "rentals", "payments", "inquiries", "chat")
    if any(table in lowered for table in personal_tables):
        if "user_id" not in lowered:
            raise ValueError("개인 테이블 조회 시 user_id 조건이 필요합니다.")

    if "fetch first" not in lowered and "limit" not in lowered:
        raw = f"{raw} FETCH FIRST 50 ROWS ONLY"
    return raw
