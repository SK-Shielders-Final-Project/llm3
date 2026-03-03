from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.clients.aws_guardrail_client import GuardrailDecision

_REFUSAL_PATTERN = re.compile(
    r"(죄송합니다|처리할 수 없|안전.*(범위|정책)|refuse|cannot comply|not allowed|blocked)",
    re.IGNORECASE,
)
_SENSITIVE_COMMAND_PATTERN = re.compile(
    r"(mkdir\s+-p\s+/mnt|ls\s+/mnt|os\.makedirs\(\s*[\"']?/mnt|os\.listdir\(\s*[\"']?/mnt|subprocess\.(run|Popen)|os\.(system|popen)\()",
    re.IGNORECASE,
)
_SENSITIVE_PATH_PATTERN = re.compile(
    r"(/(?:etc|proc|sys|dev|root|home|var|usr|bin|sbin|lib|lib64|mnt)\b)",
    re.IGNORECASE,
)
_SYSTEM_LISTING_TOKENS = {
    "boot",
    "dev",
    "etc",
    "proc",
    "sys",
    "tmp",
    "var",
    "usr",
    "bin",
    "sbin",
    "lib",
    "lib64",
    "home",
    "media",
    "mnt",
    "opt",
    "root",
    "run",
    "srv",
}
_BLOCKED_OUTPUT_MESSAGE = "보안 정책상 시스템 경로/명령 실행 결과는 제공할 수 없습니다."


class _LegacyOpenAICompatProvider:
    """openai 패키지 없이 OpenAI-compatible endpoint 호출."""

    def __init__(self, **kwargs: Any):
        kwargs.pop("type", None)
        self.model = (
            kwargs.get("model")
            or kwargs.get("model_name")
            or os.getenv("NEMO_MODEL")
            or "meta/llama-3.1-8b-instruct"
        )
        self.api_base = (
            kwargs.get("openai_api_base")
            or kwargs.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
            or "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")
        self.api_key = (
            kwargs.get("openai_api_key")
            or kwargs.get("api_key")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or ""
        )
        self.temperature = float(kwargs.get("temperature", 0.0))
        timeout_value = kwargs.get("request_timeout") or kwargs.get("timeout") or 60
        self.timeout = float(timeout_value)
        self.max_tokens = kwargs.get("max_tokens")

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs: Any) -> str:
        _ = run_manager
        payload: dict[str, Any] = {
            "model": kwargs.get("model") or self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop
        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "llm3-nemo-guardrail/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            url=f"{self.api_base}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Legacy provider HTTPError {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Legacy provider request failed: {exc}") from exc

        choices = result.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        return ""

    async def _acall(self, prompt: str, stop=None, run_manager=None, **kwargs: Any) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
        )


class NemoClient:
    def __init__(self, config_path: str):
        self._logger = logging.getLogger("guardrail_client")
        self._config_path = Path(config_path).resolve()
        self._logger.info("[NEMO-DEBUG] NemoClient.__init__ config_path=%s", self._config_path)

        self._validate_config_path()
        self._ensure_openai_compat_env()

        from nemoguardrails import LLMRails, RailsConfig
        from nemoguardrails.llm.providers import register_llm_provider

        self._register_provider(register_llm_provider)

        self._logger.info("[NEMO-DEBUG] Loading RailsConfig from %s", self._config_path)
        self.config = RailsConfig.from_path(str(self._config_path))
        self._logger.info("[NEMO-DEBUG] Initializing LLMRails")
        self.rails = LLMRails(self.config)
        self._logger.info(
            "[NEMO-DEBUG] NemoClient initialized co_files=%s",
            [p.name for p in self._co_files],
        )

    def _validate_config_path(self) -> None:
        if not self._config_path.exists() or not self._config_path.is_dir():
            raise FileNotFoundError(f"NEMO config directory not found: {self._config_path}")

        config_file = self._config_path / "config.yml"
        if not config_file.exists():
            raise FileNotFoundError(f"NEMO config.yml not found: {config_file}")

        self._co_files = sorted(self._config_path.glob("*.co"))
        if not self._co_files:
            raise FileNotFoundError(f"No .co files found in NEMO config directory: {self._config_path}")

        for co_path in self._co_files:
            try:
                content = co_path.read_text(encoding="utf-8")
            except Exception as exc:
                raise RuntimeError(f"Failed to read Colang file '{co_path.name}': {exc}") from exc
            if not content.strip():
                raise ValueError(f"Colang file is empty: {co_path.name}")

    def _ensure_openai_compat_env(self) -> None:
        if not os.getenv("OPENAI_API_KEY") and os.getenv("NVIDIA_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ["NVIDIA_API_KEY"]
            self._logger.info("[NEMO-DEBUG] OPENAI_API_KEY mapped from NVIDIA_API_KEY")

        if not os.getenv("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
            self._logger.info("[NEMO-DEBUG] OPENAI_BASE_URL defaulted to NVIDIA endpoint")

    def _register_provider(self, register_llm_provider) -> None:
        try:
            from langchain_community.chat_models import ChatOpenAI  # type: ignore
        except Exception:
            provider_cls = _LegacyOpenAICompatProvider
            provider_name = "fallback-http"
        else:
            class _ChatOpenAICompat(ChatOpenAI):
                def __init__(self, **kwargs: Any):
                    kwargs.pop("type", None)
                    super().__init__(**kwargs)

                async def _acall(self, prompt: str, stop=None, run_manager=None, **kwargs: Any) -> str:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None,
                        lambda: self._call(prompt, stop=stop, run_manager=run_manager, **kwargs),
                    )

            provider_cls = _ChatOpenAICompat
            provider_name = "langchain-chatopenai"

        try:
            register_llm_provider("legacy_openai", provider_cls)
            self._logger.info("[NEMO-DEBUG] Provider 'legacy_openai' registered (%s)", provider_name)
        except Exception as exc:
            self._logger.warning("[NEMO-DEBUG] Provider registration skipped: %s", exc)

    def apply(self, *, text: str, source: str) -> GuardrailDecision:
        if not text:
            return GuardrailDecision(action="NONE", output_text=text, raw={})

        normalized_source = (source or "INPUT").upper().strip()
        self._logger.info(
            "[NEMO-DEBUG] apply start source=%s input_len=%d",
            normalized_source,
            len(text),
        )

        if normalized_source == "OUTPUT" and self._contains_sensitive_output(text):
            self._logger.warning("[NEMO-DEBUG] BLOCK source=%s sensitive_output_precheck=True", normalized_source)
            return GuardrailDecision(
                action="BLOCK",
                output_text=_BLOCKED_OUTPUT_MESSAGE,
                raw={
                    "provider": "nemo",
                    "source": normalized_source,
                    "blocked_by_sensitive_output": True,
                    "phase": "precheck",
                },
            )

        try:
            raw_response = self._generate_with_rails(text)
            output_text = self._extract_text(raw_response).strip()
            if not output_text:
                output_text = ""

            blocked_by_refusal = bool(_REFUSAL_PATTERN.search(output_text))
            blocked_by_sensitive_output = (
                normalized_source == "OUTPUT" and self._contains_sensitive_output(output_text)
            )
            blocked = blocked_by_refusal or blocked_by_sensitive_output

            if blocked:
                self._logger.warning(
                    "[NEMO-DEBUG] BLOCK source=%s refusal=%s sensitive_output=%s",
                    normalized_source,
                    blocked_by_refusal,
                    blocked_by_sensitive_output,
                )
                blocked_text = ""
                if normalized_source == "OUTPUT":
                    blocked_text = _BLOCKED_OUTPUT_MESSAGE
                elif output_text:
                    blocked_text = output_text
                return GuardrailDecision(
                    action="BLOCK",
                    output_text=blocked_text,
                    raw={
                        "provider": "nemo",
                        "source": normalized_source,
                        "blocked_by_refusal": blocked_by_refusal,
                        "blocked_by_sensitive_output": blocked_by_sensitive_output,
                        "raw_response": raw_response,
                    },
                )

            if normalized_source == "INPUT":
                # 입력 가드레일은 차단 여부만 판단하고, 비차단 시 원문을 유지한다.
                return GuardrailDecision(
                    action="NONE",
                    output_text=text,
                    raw={"provider": "nemo", "source": normalized_source, "raw_response": raw_response},
                )

            changed = output_text.strip() != text.strip()
            action = "FILTERED" if changed else "NONE"
            return GuardrailDecision(
                action=action,
                output_text=output_text or text,
                raw={"provider": "nemo", "source": normalized_source, "raw_response": raw_response},
            )
        except Exception as exc:
            self._logger.exception("[NEMO-DEBUG] apply failed source=%s error=%s", normalized_source, exc)
            # 실패 시 fail-close: 입력은 차단, 출력은 원문 유지
            if normalized_source == "INPUT":
                return GuardrailDecision(action="BLOCK", output_text="", raw={"error": str(exc), "provider": "nemo"})
            return GuardrailDecision(action="NONE", output_text=text, raw={"error": str(exc), "provider": "nemo"})

    def _generate_with_rails(self, text: str) -> Any:
        async def _run() -> Any:
            return await self.rails.generate_async(messages=[{"role": "user", "content": text}])

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())

        if loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, _run()).result()
        return loop.run_until_complete(_run())

    def _extract_text(self, raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            content = raw.get("content")
            if isinstance(content, str):
                return content
            messages = raw.get("messages")
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, dict):
                    msg_content = last.get("content")
                    if isinstance(msg_content, str):
                        return msg_content
        return str(raw)

    def _contains_sensitive_output(self, text: str) -> bool:
        if not text:
            return False
        if _SENSITIVE_COMMAND_PATTERN.search(text):
            return True

        has_sensitive_path = bool(_SENSITIVE_PATH_PATTERN.search(text))
        if not has_sensitive_path:
            return False

        lowered = text.lower()
        token_hits = sum(1 for token in _SYSTEM_LISTING_TOKENS if token in lowered)
        # 파일 시스템 결과처럼 보이는 경우(경로 + 다수의 루트 디렉토리 토큰)만 차단한다.
        if token_hits >= 5:
            return True

        if "디렉토리 생성 성공" in text and ("파일/디렉토리 목록" in text or "목록" in text):
            return True

        return False


def build_nemo_client() -> NemoClient | None:
    logger = logging.getLogger("guardrail_client")
    logger.info("[NEMO-DEBUG] build_nemo_client() called")

    current_file_dir = Path(__file__).resolve().parent
    nemo_config_dir = current_file_dir / "NEMO"

    if not nemo_config_dir.exists():
        logger.error("[NEMO-DEBUG] Config directory NOT FOUND: %s", nemo_config_dir)
        return None

    try:
        return NemoClient(str(nemo_config_dir))
    except Exception as exc:
        logger.error("[NEMO-DEBUG] Exception during NemoClient construction: %s", exc)
        logger.exception("NemoClient init failed")
        return None

