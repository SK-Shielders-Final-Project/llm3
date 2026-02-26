from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from app.clients.aws_guardrail_client import GuardrailDecision

# ──────────────────────────────────────────────────────────────────────
# NVIDIA NeMo Guardrails Content Safety 클라이언트
# 공식 모델: nvidia/llama-3.1-nemoguard-8b-content-safety
# 참고: https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety
# ──────────────────────────────────────────────────────────────────────

# Aegis 2.0 Safety Taxonomy — 23개 카테고리
_SAFETY_TAXONOMY = (
    "S1: Violence. "
    "S2: Sexual. "
    "S3: Criminal Planning/Confessions. "
    "S4: Guns and Illegal Weapons. "
    "S5: Controlled/Regulated Substances. "
    "S6: Suicide and Self Harm. "
    "S7: Sexual (minor). "
    "S8: Hate/Identity Hate. "
    "S9: PII/Privacy. "
    "S10: Harassment. "
    "S11: Threat. "
    "S12: Profanity. "
    "S13: Needs Caution. "
    "S14: Other. "
    "S15: Manipulation. "
    "S16: Fraud/Deception. "
    "S17: Malware. "
    "S18: High Risk Gov Decision Making. "
    "S19: Political/Misinformation/Conspiracy. "
    "S20: Copyright/Trademark/Plagiarism. "
    "S21: Unauthorized Advice. "
    "S22: Illegal Activity. "
    "S23: Immoral/Unethical."
)

_INSTRUCTION = (
    "Task: Check if there is unsafe content in the user or agent messages "
    "in the below conversation in accordance with our safety policy with "
    "the below categories. Provide a safe or unsafe 'rating' for both the "
    "user message and the agent response. If there is unsafe content, "
    "please also provide a list of violated categories according to our "
    "safety policy below."
)

_OUTPUT_FORMAT = (
    'Provide your safety assessment for the above conversation in the '
    'below output JSON format: '
    '{"User Safety": safe or unsafe rating for the user message, '
    '"Response Safety": safe or unsafe rating for the agent response. '
    'Omit if no agent response present. '
    '"Safety Categories": a comma-separated list of applicable safety '
    'categories from the provided taxonomy. Omit if all safe.} '
    'Do not include anything other than the output JSON in your response. '
    'Output JSON:'
)


class NemoGuardrailClient:
    """NVIDIA NeMo Guardrails Content Safety 클라이언트.

    NVIDIA NIM API를 통해 ``nvidia/llama-3.1-nemoguard-8b-content-safety``
    모델을 호출하여 입력/출력 텍스트에 대한 콘텐츠 안전성 검사를 수행합니다.
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

        Args:
            text: 검사할 텍스트
            source: ``"INPUT"`` (사용자 입력) 또는 ``"OUTPUT"`` (LLM 응답)

        Returns:
            ``GuardrailDecision(action, output_text, raw)``
        """
        if not text:
            return GuardrailDecision(action="NONE", output_text=text, raw={})

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
        """공식 NeMo Content Safety 프롬프트 템플릿을 구성합니다.

        참고: https://docs.api.nvidia.com/nim/reference/
              nvidia-llama-3_1-nemoguard-8b-content-safety
        """
        is_output = source.upper() == "OUTPUT"

        conversation_block = ""
        if is_output:
            # LLM 응답 검사 시 — agent 메시지로 전달
            conversation_block = f"response: agent: {text}"
        else:
            # 사용자 입력 검사 시
            conversation_block = f"user: {text}"

        return (
            f"{_INSTRUCTION} "
            f"<BEGIN UNSAFE CONTENT CATEGORIES> "
            f"{_SAFETY_TAXONOMY} "
            f"<END UNSAFE CONTENT CATEGORIES> "
            f"<BEGIN CONVERSATION> "
            f"{conversation_block} "
            f"<END CONVERSATION> "
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
    import re

    json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None
