from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

from app.clients.aws_guardrail_client import GuardrailDecision

# ──────────────────────────────────────────────────────────────────────
# NVIDIA NeMo Guardrails Content Safety 클라이언트
# 공식 모델: nvidia/llama-3.1-nemoguard-8b-content-safety
# 참고: https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety
# ──────────────────────────────────────────────────────────────────────

# ── 특수 토큰 주입(Turn Injection) 탐지 ──────────────────────────────
# Gemma 3 등 모델의 특수 토큰을 이용한 턴 강제 종료/주입 공격을 사전 차단.
# NeMo Content Safety 모델은 콘텐츠 카테고리만 검사하므로, 이런 구조적
# prompt injection은 별도 로컬 필터가 필요하다.
_SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\s*/?(?:"
    r"end_of_turn|start_of_turn|eos|bos|pad|unk|"           # Gemma 계열
    r"start_header_id|end_header_id|eot_id|"                  # Llama 계열
    r"\|(?:im_start|im_end|system|user|assistant)\|"          # ChatML 계열
    r")\s*>",
    re.IGNORECASE,
)

# Aegis 2.0 Safety Taxonomy — 23개 카테고리 (한국어)
_SAFETY_TAXONOMY = (
    "S1: 폭력. "
    "S2: 성적 콘텐츠. "
    "S3: 범죄 계획/고백. "
    "S4: 총기 및 불법 무기. "
    "S5: 규제 약물/물질. "
    "S6: 자살 및 자해. "
    "S7: 미성년자 성적 콘텐츠. "
    "S8: 혐오/차별 발언. "
    "S9: 개인정보/프라이버시 침해. "
    "S10: 괴롭힘. "
    "S11: 위협. "
    "S12: 욕설/비속어. "
    "S13: 주의 필요. "
    "S14: 기타. "
    "S15: 조작/심리 조종. "
    "S16: 사기/기만. "
    "S17: 악성코드/멀웨어. "
    "S18: 고위험 정부 의사결정. "
    "S19: 정치적 허위정보/음모론. "
    "S20: 저작권/상표권/표절. "
    "S21: 무허가 조언. "
    "S22: 불법 활동. "
    "S23: 비도덕적/비윤리적 행위."
)

_INSTRUCTION = (
    "작업: 아래 대화에서 사용자 또는 에이전트 메시지에 안전 정책에 "
    "위반되는 위험한 콘텐츠가 있는지 확인하세요. 아래 카테고리를 기준으로 "
    "사용자 메시지와 에이전트 응답 각각에 대해 'safe' 또는 'unsafe' 등급을 "
    "부여하세요. 위험한 콘텐츠가 있는 경우, 위반된 카테고리 목록도 함께 "
    "제공하세요."
)

_OUTPUT_FORMAT = (
    '위 대화에 대한 안전성 평가를 아래 JSON 형식으로 제공하세요: '
    '{"User Safety": 사용자 메시지에 대한 safe 또는 unsafe 등급, '
    '"Response Safety": 에이전트 응답에 대한 safe 또는 unsafe 등급. '
    '에이전트 응답이 없으면 생략. '
    '"Safety Categories": 위반된 안전 카테고리의 쉼표로 구분된 목록. '
    '모두 안전하면 생략.} '
    '출력 JSON 외에 다른 내용은 포함하지 마세요. '
    '출력 JSON:'
)


class NemoGuardrailClient:
    """NVIDIA NeMo Guardrails Content Safety 클라이언트.

    NVIDIA NIM API를 통해 ``nvidia/llama-3.1-nemoguard-8b-content-safety``
    모델을 호출하여 입력/출력 텍스트에 대한 콘텐츠 안전성 검사를 수행합니다.

    추가로, LLM 특수 토큰(``<end_of_turn>``, ``<start_of_turn>`` 등)을
    이용한 턴 주입(prompt injection) 공격은 NeMo API 호출 전에 로컬에서
    사전 차단합니다.
    """

    MODEL = "nvidia/llama-3.1-nemoguard-8b-content-safety"
    DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str | None = None,
        timeout_seconds: int = 15,
    ) -> None:
        self._api_key = api_key
        self._endpoint = (endpoint or self.DEFAULT_ENDPOINT).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._logger = logging.getLogger("guardrail_client")

    # ── GuardrailClientProtocol 구현 ─────────────────────────────────
    def apply(self, *, text: str, source: str) -> GuardrailDecision:
        """텍스트에 대해 NeMo Content Safety 검사를 수행합니다.

        1단계: 특수 토큰 주입(Turn Injection) 사전 차단 (로컬)
        2단계: NeMo Content Safety 모델 API 호출 (원격)

        Args:
            text: 검사할 텍스트
            source: ``"INPUT"`` (사용자 입력) 또는 ``"OUTPUT"`` (LLM 응답)

        Returns:
            ``GuardrailDecision(action, output_text, raw)``
        """
        if not text:
            return GuardrailDecision(action="NONE", output_text=text, raw={})

        # ── 1단계: 특수 토큰 주입 사전 차단 ──────────────────────────
        token_match = _SPECIAL_TOKEN_PATTERN.search(text)
        if token_match:
            matched = token_match.group()
            self._logger.warning(
                "NeMo Guardrail BLOCKED — 특수 토큰 주입 감지 "
                "source=%s matched_token=%s input_len=%d",
                source,
                matched,
                len(text),
            )
            return GuardrailDecision(
                action="BLOCK",
                output_text="",
                raw={"blocked_by": "special_token_filter", "matched": matched},
            )

        prompt = self._build_prompt(text, source)

        payload: dict[str, Any] = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 128,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "LLM-SecLab/1.0",
        }
        request = urllib.request.Request(
            url=self._endpoint,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._logger.error(
                "NeMo Guardrail HTTPError status=%s detail=%s",
                exc.code,
                detail,
            )
            raise
        except Exception:
            self._logger.exception(
                "NeMo Guardrail apply failed endpoint=%s", self._endpoint
            )
            raise

        action, output_text = _extract_nemo_result(
            raw_response, original_text=text, source=source
        )
        changed = (output_text or "").strip() != text.strip()
        self._logger.info(
            "NeMo Guardrail applied source=%s action=%s changed=%s "
            "input_len=%d output_len=%d",
            source,
            action,
            changed,
            len(text),
            len(output_text or ""),
        )
        return GuardrailDecision(
            action=action, output_text=output_text, raw=raw_response
        )

    # ── 프롬프트 구성 ────────────────────────────────────────────────
    @staticmethod
    def _build_prompt(text: str, source: str) -> str:
        """공식 NeMo Content Safety 프롬프트 템플릿을 한국어로 구성합니다."""
        is_output = source.upper() == "OUTPUT"

        conversation_block = ""
        if is_output:
            # LLM 응답 검사 시 — 에이전트 메시지로 전달
            conversation_block = f"응답: 에이전트: {text}"
        else:
            # 사용자 입력 검사 시
            conversation_block = f"사용자: {text}"

        return (
            f"{_INSTRUCTION} "
            f"<위험 콘텐츠 카테고리 시작> "
            f"{_SAFETY_TAXONOMY} "
            f"<위험 콘텐츠 카테고리 끝> "
            f"<대화 시작> "
            f"{conversation_block} "
            f"<대화 끝> "
            f"{_OUTPUT_FORMAT}"
        )


# ── 팩토리 ────────────────────────────────────────────────────────────
def build_nemo_guardrail_client_from_env() -> NemoGuardrailClient | None:
    """환경 변수에서 NeMo Guardrail 클라이언트를 생성합니다."""
    enabled = (
        os.getenv("NEMO_GUARDRAIL_ENABLED", "false").strip().lower()
        in {"1", "true", "yes", "y", "on"}
    )
    if not enabled:
        logging.getLogger("guardrail_client").info("NeMo Guardrail disabled")
        return None

    api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        logging.getLogger("guardrail_client").warning(
            "NeMo Guardrail enabled but NVIDIA_API_KEY is missing"
        )
        return None

    endpoint = (
        os.getenv("NEMO_GUARDRAIL_ENDPOINT")
        or NemoGuardrailClient.DEFAULT_ENDPOINT
    ).strip()
    timeout_raw = (os.getenv("NEMO_TIMEOUT_SECONDS") or "15").strip()
    try:
        timeout_seconds = int(timeout_raw)
    except ValueError:
        timeout_seconds = 15

    logging.getLogger("guardrail_client").info(
        "NeMo Guardrail configured enabled=true endpoint=%s timeout=%s",
        endpoint,
        timeout_seconds,
    )
    return NemoGuardrailClient(
        api_key=api_key,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
    )


# ── 응답 파싱 ─────────────────────────────────────────────────────────
def _extract_nemo_result(
    response: dict[str, Any],
    *,
    original_text: str,
    source: str,
) -> tuple[str, str]:
    """NeMo Content Safety 모델의 JSON 응답을 파싱합니다.

    모델 응답 형식 (JSON):
        {
            "User Safety": "safe" | "unsafe",
            "Response Safety": "safe" | "unsafe",        // 선택
            "Safety Categories": "Violence, Profanity"   // unsafe일 때만
        }
    """
    logger = logging.getLogger("guardrail_client")

    if not isinstance(response, dict):
        return "NONE", original_text

    # NIM chat completions 응답에서 텍스트 추출
    choices = response.get("choices") or []
    content = ""
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message") or {}
            if isinstance(message, dict):
                content = message.get("content", "")

    if not content:
        return "NONE", original_text

    # JSON 파싱 시도
    safety_json = _parse_safety_json(content)
    if safety_json is None:
        logger.warning(
            "NeMo Guardrail: 모델 응답 JSON 파싱 실패 content=%s",
            content[:200],
        )
        return "NONE", original_text

    # source에 따라 적절한 safety 필드 확인
    is_output = source.upper() == "OUTPUT"
    if is_output:
        verdict = (safety_json.get("Response Safety") or "safe").strip().lower()
    else:
        verdict = (safety_json.get("User Safety") or "safe").strip().lower()

    if verdict == "safe":
        return "NONE", original_text

    if verdict == "unsafe":
        categories = safety_json.get("Safety Categories", "unknown")
        logger.warning(
            "NeMo Guardrail BLOCKED content source=%s categories=%s",
            source,
            categories,
        )
        return "BLOCK", ""

    # 판별 불가 시 안전하다고 간주
    return "NONE", original_text


def _parse_safety_json(content: str) -> dict[str, Any] | None:
    """모델 응답에서 JSON을 추출합니다."""
    content = content.strip()

    # 직접 JSON 파싱
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # ```json 블록 안에 JSON이 있을 수 있음

    json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None
