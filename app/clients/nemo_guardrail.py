from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from app.clients.aws_guardrail_client import GuardrailDecision


class NemoGuardrailClient:
    """NVIDIA NeMo Guardrails 클라이언트 (NIM API 기반 콘텐츠 안전 검사)."""

    MODEL = "nvidia/llama-3.1-nemoguard-8b-content-safety"
    DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

    # NeMo Content Safety가 검사하는 카테고리 목록
    SAFETY_CATEGORIES = (
        "Violence, Hate Speech, Sexual Content, Self-Harm, "
        "Harassment, Threat, Deception, Illegal Activities, Profanity"
    )

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

    def apply(self, *, text: str, source: str) -> GuardrailDecision:
        """
        텍스트에 대해 NeMo 콘텐츠 안전 검사를 수행합니다.

        Parameters:
            text: 검사할 텍스트
            source: "INPUT" (사용자 입력) 또는 "OUTPUT" (LLM 응답)

        Returns:
            GuardrailDecision(action, output_text, raw)
        """
        if not text:
            return GuardrailDecision(action="NONE", output_text=text, raw={})

        role = "assistant" if source.upper() == "OUTPUT" else "user"

        # NeMo Content Safety 프롬프트 구성
        safety_prompt = self._build_safety_prompt(text, role)

        payload: dict[str, Any] = {
            "model": self.MODEL,
            "messages": [
                {"role": "user", "content": safety_prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 32,
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
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._logger.error(
                "NeMo Guardrail HTTPError status=%s detail=%s", exc.code, detail
            )
            raise
        except Exception:
            self._logger.exception(
                "NeMo Guardrail apply failed endpoint=%s", self._endpoint
            )
            raise

        action, output_text = _extract_nemo_result(raw_response, original_text=text)
        changed = (output_text or "").strip() != text.strip()
        self._logger.info(
            "NeMo Guardrail applied source=%s action=%s changed=%s input_len=%d output_len=%d",
            source,
            action,
            changed,
            len(text),
            len(output_text or ""),
        )
        return GuardrailDecision(action=action, output_text=output_text, raw=raw_response)

    def _build_safety_prompt(self, text: str, role: str) -> str:
        """NeMo Content Safety 모델에 전달할 프롬프트를 구성합니다."""
        if role == "user":
            return (
                f"Task: Check if there is unsafe content in 'User' messages "
                f"in conversations according to our safety policy with the below categories.\n\n"
                f"<BEGIN UNSAFE CONTENT CATEGORIES>\n"
                f"{self.SAFETY_CATEGORIES}\n"
                f"<END UNSAFE CONTENT CATEGORIES>\n\n"
                f"<BEGIN CONVERSATION>\n"
                f"User: {text}\n"
                f"<END CONVERSATION>\n\n"
                f"Provide your safety assessment for 'User' in the above conversation:\n"
                f"- First line must read 'safe' or 'unsafe'.\n"
                f"- If unsafe, a second line must include a comma-separated list of "
                f"violated categories."
            )
        else:
            return (
                f"Task: Check if there is unsafe content in 'Assistant' messages "
                f"in conversations according to our safety policy with the below categories.\n\n"
                f"<BEGIN UNSAFE CONTENT CATEGORIES>\n"
                f"{self.SAFETY_CATEGORIES}\n"
                f"<END UNSAFE CONTENT CATEGORIES>\n\n"
                f"<BEGIN CONVERSATION>\n"
                f"Assistant: {text}\n"
                f"<END CONVERSATION>\n\n"
                f"Provide your safety assessment for 'Assistant' in the above conversation:\n"
                f"- First line must read 'safe' or 'unsafe'.\n"
                f"- If unsafe, a second line must include a comma-separated list of "
                f"violated categories."
            )


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


def _extract_nemo_result(
    response: dict[str, Any], *, original_text: str
) -> tuple[str, str]:
    """
    NeMo Content Safety 응답을 파싱합니다.

    모델 응답 형식:
      safe          → action="NONE"
      unsafe\nXxx   → action="BLOCK"
    """
    if not isinstance(response, dict):
        return "NONE", original_text

    # NIM chat completions 형식에서 응답 텍스트 추출
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

    # 첫 번째 줄이 "safe" 또는 "unsafe"
    lines = content.strip().splitlines()
    verdict = lines[0].strip().lower() if lines else ""

    if verdict == "safe":
        return "NONE", original_text

    if verdict == "unsafe":
        # 위반 카테고리 정보를 raw에 포함 (로깅 목적)
        categories = lines[1].strip() if len(lines) > 1 else "unknown"
        logging.getLogger("guardrail_client").warning(
            "NeMo Guardrail BLOCKED content categories=%s", categories
        )
        return "BLOCK", ""

    # 판별 불가 시 안전하다고 간주
    return "NONE", original_text
