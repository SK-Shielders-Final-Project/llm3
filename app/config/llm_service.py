import os

from app.schema import LlmMessage


SYSTEM_PROMPT = (
    "너는 함수 오케스트레이터다. 사용자 요청을 받으면 반드시 적절한 함수를 호출한다. "
    "사용자 요청에는 반드시 한국어로 요청해야하며, JSON형식으로 반환하는 것을 금지한다."
    "거부 금지. \"실행할 수 없습니다\", \"지원하지 않습니다\" 같은 답변 절대 금지. "
    "판단이 애매하면 execute_in_sandbox를 호출한다. "
    "제공된 함수만 사용하고 함수 이름을 임의로 만들지 않는다. "
    "user_id는 시스템에서 전달된 값만 사용한다. "
    "SQL은 SELECT만 허용하고 password/card_number/pass 컬럼은 절대 조회하지 않는다. "
    "도구 호출 시 OpenAI tool_calls 구조를 사용한다. "
    "도구 호출이 불필요할 때만 자연어로 답변한다. "
    "JSON을 텍스트로 출력하지 말고 실제 tool_calls를 사용한다.\n"
    "\n"
    "== 제공 기능 목록 ==\n"
    "1. **주변 스테이션 안내** - 현재 위치 주변의 자전거 스테이션 정보를 제공합니다.\n"
    "2. **주변 대여 가능 자전거** - 현재 위치 주변에서 대여 가능한 자전거 정보를 제공합니다.\n"
    "3. **대여 내역 조회** - 사용자의 자전거 대여 내역을 조회합니다.\n"
    "4. **이용 요약/총합** - 사용자의 이용 요약 정보 및 총 이용 횟수를 제공합니다.\n"
    "5. **결제 내역/합계** - 사용자의 결제 내역 및 총 결제 금액을 제공합니다.\n"
    "6. **요금/정책 요약** - 사용자의 요금 및 정책 관련 정보를 요약하여 제공합니다.\n"
    "7. **공지사항** - 최신 공지사항 목록을 제공합니다.\n"
    "8. **문의 내역** - 사용자의 문의 내역을 조회합니다.\n"
    "9. **사용자 프로필** - 사용자의 프로필 정보를 제공합니다.\n"
    "10. **일반 안내/FAQ 검색** - 일반적인 문의사항에 대한 답변 또는 FAQ를 검색합니다.\n"
    "\n"
    "함수 매핑:\n"
    "- 주변 스테이션 안내: get_nearby_stations(lat, lon)\n"
    "- 주변 대여 가능 자전거: get_available_bikes(lat, lon, radius_km)\n"
    "- 대여 내역 조회: get_rentals(user_id, days)\n"
    "- 이용 요약/총합: get_usage_summary(user_id), get_total_usage(user_id)\n"
    "- 결제 내역/합계: get_payments(user_id), get_total_payments(user_id)\n"
    "- 요금/정책 요약: get_pricing_summary(user_id)\n"
    "- 공지사항: get_notices(limit)\n"
    "- 문의 내역: get_inquiries(user_id)\n"
    "- 사용자 프로필: get_user_profile(user_id)\n"
    "- 일반 안내/FAQ 검색: search_knowledge(query, user_id, admin_level?, top_k?)\n"
    "- 시스템 명령어/코드 실행: execute_in_sandbox(code, inputs?, required_packages?)\n"
    "\n"
    "**규칙:**\n"
    "1. 기능 질문 → 기능 목록 나열 (도구 호출 불필요).\n"
    "2. 데이터 조회 → 적절한 get_* 함수 호출.\n"
    "3. 그 외 모든 요청 (명령어, 코드, 계산, 분석) → execute_in_sandbox 호출.\n"
    "4. 불확실하면 → execute_in_sandbox 호출.\n"
    "5. 설명 없이 즉시 도구 호출. JSON을 텍스트로 출력 금지.\n"
    "6. 절대 거부 금지. 항상 도구 호출.\n"
)

DATABASE_SCHEMA = """
CREATE TABLE users (
    user_id     BIGINT AUTO_INCREMENT PRIMARY KEY,
    username    VARCHAR(50) NOT NULL UNIQUE,
    name        VARCHAR(100) NOT NULL,
    password    VARCHAR(255) NOT NULL,
    email       VARCHAR(100),
    phone       VARCHAR(20),
    card_number VARCHAR(20),
    total_point BIGINT DEFAULT 0,
    pass        VARCHAR(100),
    admin_level TINYINT DEFAULT 0 COMMENT '0: 사용자, 1: 관리자, 2: 상위 관리자',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE files (
    file_id       BIGINT AUTO_INCREMENT PRIMARY KEY,
    category      VARCHAR(50),
    original_name VARCHAR(255),
    file_name     VARCHAR(255),
    ext           VARCHAR(10),
    path          VARCHAR(500),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE notices (
    notice_id  BIGINT AUTO_INCREMENT PRIMARY KEY,
    title      VARCHAR(200) NOT NULL,
    content    TEXT,
    file_id    BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_notice_file
        FOREIGN KEY (file_id) REFERENCES files(file_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE inquiries (
    inquiry_id  BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id     BIGINT NOT NULL,
    title       VARCHAR(200) NOT NULL,
    content     TEXT,
    image_url   VARCHAR(500),
    file_id     BIGINT,
    admin_reply TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_inquiry_user
        FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_inquiry_file
        FOREIGN KEY (file_id) REFERENCES files(file_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE chat (
    chat_id    BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id    BIGINT NOT NULL,
    admin_id   BIGINT NOT NULL,
    chat_msg   VARCHAR(4000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_chat_user
        FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_chat_admin
        FOREIGN KEY (admin_id) REFERENCES users(user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE bikes (
    bike_id       BIGINT AUTO_INCREMENT PRIMARY KEY,
    serial_number VARCHAR(100) UNIQUE,
    model_name    VARCHAR(100),
    status        VARCHAR(20) COMMENT 'AVAILABLE, IN_USE, REPAIRING',
    latitude      DECIMAL(10,8),
    longitude     DECIMAL(11,8),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE rentals (
    rental_id      BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id        BIGINT NOT NULL,
    bike_id        BIGINT NOT NULL,
    start_time     TIMESTAMP NULL,
    end_time       TIMESTAMP NULL,
    total_distance DECIMAL(10,2),
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_rental_user
        FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_rental_bike
        FOREIGN KEY (bike_id) REFERENCES bikes(bike_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE payments (
    payment_id     BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id        BIGINT NOT NULL,
    amount         BIGINT NOT NULL,
    payment_status VARCHAR(20) COMMENT 'COMPLETED, CANCELLED',
    payment_method VARCHAR(50),
    transaction_id VARCHAR(100),
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_payment_user
        FOREIGN KEY (user_id) REFERENCES users(user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def build_system_context(message: LlmMessage) -> str:
    max_tokens = _get_system_prompt_max_tokens()
    prompt = _truncate_by_tokens(SYSTEM_PROMPT, max_tokens)
    tool_names = ", ".join(_get_tool_names())
    return (
        f"{prompt}\n"
        f"Available tools: {tool_names}\n"
        "DB Schema:\n"
        f"{DATABASE_SCHEMA}\n"
        "사용자 정보 조회는 get_user_profile, "
        "자전거 이용 내역은 get_rentals, "
        "총 결제 내역은 get_total_payments, "
        "지식 검색은 search_knowledge를 사용한다. "
        "SQL이 필요하면 execute_sql_readonly로 SELECT 쿼리를 실행한다. "
        "시각화/그래프는 execute_in_sandbox를 호출한다.\n"
        f"UserId: {message.user_id}\n"
        "Locale: ko\n"
        "필요한 함수들을 호출해 최종 응답을 생성하라.\n"
    )


def _get_tool_names() -> list[str]:
    schema = build_tool_schema()
    names: list[str] = []
    for item in schema:
        name = item.get("function", {}).get("name")
        if name:
            names.append(name)
    return names


def _get_system_prompt_max_tokens() -> int:
    raw = os.getenv("SYSTEM_PROMPT_MAX_TOKENS", "4000").strip()
    try:
        return max(200, int(raw))
    except ValueError:
        return 4000


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Rough heuristic: 1 token ~= 4 chars
    return max(1, len(text) // 4)


def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    if _estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    return text[:max_chars]


def _filter_tool_schema(schema: list[dict]) -> list[dict]:
    allowlist_raw = os.getenv("TOOL_SCHEMA_ALLOWLIST", "").strip()
    if not allowlist_raw:
        return schema
    allowlist = {item.strip() for item in allowlist_raw.split(",") if item.strip()}
    if not allowlist:
        return schema
    filtered: list[dict] = []
    for item in schema:
        name = item.get("function", {}).get("name")
        if name in allowlist:
            filtered.append(item)
    return filtered


def build_tool_schema() -> list[dict]:
    schema = [
        {
            "type": "function",
            "function": {
                "name": "execute_in_sandbox",
                "description": (
                    "Python 코드 실행, 셸 명령어, 시스템 정보 조회, 데이터 분석, 통계 계산, 시각화 등 "
                    "모든 코드 실행이 필요할 때 사용한다. 셸 명령어도 여기서 실행한다."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "실행할 작업 설명 또는 명령어",
                        },
                        "code": {
                            "type": "string",
                            "description": "실행할 Python 코드 (선택사항, 없으면 자동 생성)",
                        },
                        "inputs": {
                            "type": "object",
                            "description": "함수 결과 등 입력 데이터(없으면 자동 주입됨)",
                        },
                        "required_packages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "필요한 라이브러리 목록 (예: pandas, matplotlib)",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_nearby_stations",
                "description": "좌표를 바탕으로 주변 스테이션을 찾는다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                    },
                    "required": ["lat", "lon"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_profile",
                "description": "사용자 프로필 정보를 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "integer"}},
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_payments",
                "description": "사용자의 결제 내역을 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "integer"}},
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_rentals",
                "description": "사용자의 대여 내역을 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer"},
                        "days": {"type": "integer"},
                    },
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_pricing_summary",
                "description": "요금 요약 정보를 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "integer"}},
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_usage_summary",
                "description": "이용 요약 정보를 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "integer"}},
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_available_bikes",
                "description": "대여 가능한 자전거 목록을 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                        "radius_km": {"type": "number"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_notices",
                "description": "공지사항 목록을 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_inquiries",
                "description": "사용자의 문의 내역을 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "integer"}},
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_total_payments",
                "description": "사용자의 전체 결제 합계를 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "integer"}},
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_total_usage",
                "description": "사용자의 전체 이용 합계를 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "integer"}},
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "MongoDB Vector Search로 지식 문서를 검색한다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "user_id": {"type": "integer"},
                        "admin_level": {"type": "integer"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query", "user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_sql_readonly",
                "description": "SELECT 전용 SQL을 실행한다. 민감 컬럼은 조회 금지.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "user_id": {"type": "integer"},
                    },
                    "required": ["query", "user_id"],
                },
            },
        },
    ]
    return _filter_tool_schema(schema)