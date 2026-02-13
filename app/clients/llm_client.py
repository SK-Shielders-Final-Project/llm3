from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LlmResponse:
    content: str | None
    tool_calls: list[ToolCall]
    model: str


class LlmClient:
    """
    실제 LLM SDK를 감싸는 최소 어댑터.
    외부에서 호출 함수를 주입받아 사용한다.
    """

    def __init__(self, completion_func: Callable[[list[dict], list[dict]], Any]) -> None:
        self._completion_func = completion_func

    def create_completion(self, messages: list[dict], tools: list[dict]) -> LlmResponse:
        raw = self._completion_func(messages, tools)
        return normalize_response(raw)


def build_http_completion_func() -> Callable[[list[dict], list[dict]], Any]:
    base_url = os.getenv("LLM_BASE_URL")
    if not base_url:
        raise RuntimeError("LLM_BASE_URL이 설정되지 않았습니다.")

    model = os.getenv("MODEL_ID", "RedHatAI/gemma-3-27b-it-quantized.w4a16")
    temperature_raw = os.getenv("TEMPERATURE", "0.7")
    top_p_raw = os.getenv("TOP_P", "0.9")
    max_tokens_raw = os.getenv("MAX_TOKENS", "1024")
    max_model_len_raw = os.getenv("MAX_MODEL_LEN", "8192")
    ctx_buffer_raw = os.getenv("LLM_CONTEXT_BUFFER_TOKENS", "64")
    timeout_raw = os.getenv("LLM_TIMEOUT_SECONDS", "20")

    temperature = float(temperature_raw) if temperature_raw.strip() else 0.7
    top_p = float(top_p_raw) if top_p_raw.strip() else 0.9
    max_tokens = int(max_tokens_raw) if max_tokens_raw.strip() else 1024
    max_model_len = int(max_model_len_raw) if max_model_len_raw.strip() else 8192
    ctx_buffer = int(ctx_buffer_raw) if ctx_buffer_raw.strip() else 64
    timeout_seconds = int(timeout_raw) if timeout_raw.strip() else 20
    
    # Unbounded Consumption 취약점: 토큰/타임아웃 제한 완화
    vulnerable_unbounded = os.getenv("VULNERABLE_UNBOUNDED_CONSUMPTION", "false").strip().lower() in {"true", "1", "yes"}
    if vulnerable_unbounded:
        max_tokens = max_model_len  # 컨텍스트 전체를 출력 토큰으로 허용 → VRAM 폭증 유발
        timeout_seconds = 600  # 10분 타임아웃 (긴 생성 허용)

    base_url = base_url.rstrip("/")
    endpoint = os.getenv("LLM_CHAT_ENDPOINT", f"{base_url}/chat/completions")
    api_key = os.getenv("LLM_API_KEY")
    logger = logging.getLogger("llm_client")
    log_verbose = os.getenv("LLM_LOG_VERBOSE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    use_gemma_message_adapter = os.getenv("USE_GEMMA_MESSAGE_ADAPTER", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    def _completion(messages: list[dict], tools: list[dict]) -> Any:
        start = time.monotonic()
        prepared_messages = _prepare_messages_for_request(
            messages=messages,
            model=model,
            use_gemma_adapter=use_gemma_message_adapter,
        )

        def _estimate_input_tokens(msgs: list[dict], tool_schema: list[dict]) -> int:
            """
            vLLM/OpenAI 호환 서버의 'input tokens'는 messages + tools를 모두 포함한다.
            정확한 토크나이저가 없으니, JSON 직렬화 길이를 기반으로 대략 추정한다.
            """
            try:
                blob = json.dumps({"messages": msgs, "tools": tool_schema}, ensure_ascii=False)
            except Exception:
                blob = str({"messages": msgs, "tools": tool_schema})
            # 대략 4 chars ~= 1 token 가정 + 턴/포맷 오버헤드 보정
            return max(1, (len(blob) // 4) + (8 * max(1, len(msgs))))

        est_input_tokens = _estimate_input_tokens(prepared_messages, tools or [])
        available = max_model_len - est_input_tokens - ctx_buffer
        desired_max_tokens = max_tokens

        # Unbounded Consumption 취약점: 토큰 캡핑 우회 → VRAM 고갈 가능
        if vulnerable_unbounded:
            capped_max_tokens = desired_max_tokens  # 캡핑 없이 그대로 사용
            logger.warning(
                "[VULN] Unbounded Consumption 활성: 토큰 캡핑 우회 desired=%s (캡핑 없음)",
                desired_max_tokens,
            )
        else:
            capped_max_tokens = max(16, min(desired_max_tokens, max(16, available)))

        payload: dict[str, Any] = {
            "model": model,
            "messages": prepared_messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": capped_max_tokens,
            "stream": False,
        }
        logger.info(
            "LLM max_tokens 캡핑 desired=%s capped=%s max_ctx=%s est_input=%s buffer=%s",
            desired_max_tokens,
            capped_max_tokens,
            max_model_len,
            est_input_tokens,
            ctx_buffer,
        )
        if log_verbose:
            safe_messages = _sanitize_messages(prepared_messages)
            tool_names = _extract_tool_names(tools or [])
            logger.info(
                "LLM 요청 전송 messages=%s tools=%s endpoint=%s",
                json.dumps(safe_messages, ensure_ascii=False),
                json.dumps(tool_names, ensure_ascii=False),
                endpoint,
            )
        else:
            logger.info(
                "LLM 요청 전송 message_count=%d tool_count=%d endpoint=%s",
                len(prepared_messages),
                len(tools or []),
                endpoint,
            )
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            url=endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
                elapsed = time.monotonic() - start
                logger.info(
                    "LLM 응답 성공 elapsed=%.2fs tool_count=%s endpoint=%s",
                    elapsed,
                    len(tools or []),
                    endpoint,
                )
                if log_verbose:
                    logger.info("LLM raw 응답=%s", json.dumps(data, ensure_ascii=False))
                return data
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            tool_count = len(tools or [])
            tool_hint = "tools" if tool_count > 0 else "no-tools"
            logger.error(
                "LLM 요청 실패(%s) tool_count=%s tool_hint=%s endpoint=%s detail=%s",
                exc.code,
                tool_count,
                tool_hint,
                endpoint,
                detail,
            )

            # vLLM/OpenAI 호환 서버에서 max_tokens가 컨텍스트 길이를 초과하면 400으로 떨어진다.
            # 에러 메시지에서 max_ctx/input_tokens를 파싱해 자동으로 낮춰 1회 재시도한다.
            if exc.code == 400 and "maximum context length" in detail.lower():
                # 예: "'max_tokens' ... maximum context length is 8192 tokens and your request has 405 input tokens ..."
                ctx_match = re.search(
                    r"maximum context length is\s+(\d+)\s+tokens\s+and your request has\s+(\d+)\s+input tokens",
                    detail,
                    flags=re.IGNORECASE,
                )
                param_match = re.search(r"parameter=(max_tokens|max_completion_tokens)", detail, flags=re.IGNORECASE)
                if ctx_match and param_match:
                    max_ctx = int(ctx_match.group(1))
                    input_tokens = int(ctx_match.group(2))
                    param_name = param_match.group(1)
                    # 약간의 버퍼를 남겨서 다시 실패하지 않게 한다.
                    new_max = max(16, max_ctx - input_tokens - 8)
                    old_max = payload.get(param_name) or payload.get("max_tokens")
                    payload[param_name] = new_max
                    logger.warning(
                        "max_tokens 자동 조정 후 재시도 param=%s old=%s new=%s max_ctx=%s input=%s",
                        param_name,
                        old_max,
                        new_max,
                        max_ctx,
                        input_tokens,
                    )
                    retry_data = json.dumps(payload).encode("utf-8")
                    retry_request = urllib.request.Request(
                        url=endpoint,
                        data=retry_data,
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(retry_request, timeout=timeout_seconds) as response:
                        retry_json = json.loads(response.read().decode("utf-8"))
                        elapsed = time.monotonic() - start
                        logger.info(
                            "LLM 응답 성공(retry) elapsed=%.2fs tool_count=%s endpoint=%s",
                            elapsed,
                            len(tools or []),
                            endpoint,
                        )
                        logger.info("LLM raw 응답(retry)=%s", json.dumps(retry_json, ensure_ascii=False))
                        return retry_json

            if "roles must alternate" in detail.lower():
                flattened = _flatten_messages(prepared_messages)
                est_input_tokens2 = _estimate_input_tokens(flattened, tools or [])
                available2 = max_model_len - est_input_tokens2 - ctx_buffer
                capped2 = max(16, min(desired_max_tokens, max(16, available2)))
                retry_payload = {
                    "model": model,
                    "messages": flattened,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": capped2,
                    "stream": False,
                }
                retry_data = json.dumps(retry_payload).encode("utf-8")
                retry_request = urllib.request.Request(
                    url=endpoint,
                    data=retry_data,
                    headers=headers,
                    method="POST",
                )
                logger.warning("LLM 역할 제약 감지: 메시지 평탄화 후 재시도")
                with urllib.request.urlopen(retry_request, timeout=timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))

            if "auto\" tool choice requires" in detail:
                raise RuntimeError(
                    "LLM 서버가 tool_choice=auto를 지원하지 않습니다. "
                    "LLM 서버 실행 옵션에 --enable-auto-tool-choice 및 "
                    "--tool-call-parser를 설정하세요."
                ) from exc

            tool_error = "tools" in detail.lower() or "tool" in detail.lower()
            hint = " (tools 미지원 가능성)" if tool_error else ""
            raise RuntimeError(f"LLM 요청 실패({exc.code}){hint}: {detail}") from exc
        except urllib.error.URLError as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "LLM 요청 타임아웃/네트워크 실패 elapsed=%.2fs endpoint=%s error=%s",
                elapsed,
                endpoint,
                exc,
            )
            raise RuntimeError(f"LLM 요청 실패: {exc}") from exc

    return _completion


def _flatten_messages(messages: list[dict]) -> list[dict]:
    content = "\n".join(f"[{msg['role']}] {msg['content']}" for msg in messages)
    return [{"role": "user", "content": content}]


def _prepare_messages_for_request(
    *,
    messages: list[dict],
    model: str,
    use_gemma_adapter: bool,
) -> list[dict]:
    if not use_gemma_adapter:
        return messages
    if "gemma" not in (model or "").lower():
        return messages
    use_raw_turn_prompt = os.getenv("USE_RAW_GEMMA_TURN_PROMPT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if use_raw_turn_prompt:
        return _build_raw_gemma_turn_prompt(messages)
    return _adapt_messages_for_gemma_template(messages)


def _adapt_messages_for_gemma_template(messages: list[dict]) -> list[dict]:
    """
    Gemma chat_template 규칙과 맞추기 위한 전처리:
    1) 첫 system 메시지를 첫 user 메시지 prefix로 이동
    2) 대화 role을 user/assistant 교대로 보정
    """
    if not messages:
        return []

    copied = [{"role": msg.get("role"), "content": msg.get("content", "")} for msg in messages]
    system_prefix = ""
    start_index = 0

    if copied and copied[0].get("role") == "system":
        system_prefix = _content_to_text(copied[0].get("content"))
        start_index = 1

    body = copied[start_index:]
    if not body:
        return [{"role": "user", "content": system_prefix.strip()}]

    if system_prefix:
        first = body[0]
        if first.get("role") == "user":
            first_text = _content_to_text(first.get("content"))
            first["content"] = f"{system_prefix.strip()}\n\n{first_text}".strip()
        else:
            body.insert(0, {"role": "user", "content": system_prefix.strip()})

    normalized: list[dict] = []
    expected_role = "user"
    for msg in body:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            role = "user"
        content = msg.get("content", "")
        if not normalized:
            if role != expected_role:
                normalized.append({"role": expected_role, "content": ""})
            normalized.append({"role": role, "content": content})
            expected_role = "assistant" if role == "user" else "user"
            continue

        prev = normalized[-1]
        if role == prev.get("role"):
            prev_text = _content_to_text(prev.get("content"))
            content_text = _content_to_text(content)
            prev["content"] = f"{prev_text}\n\n{content_text}".strip()
            continue

        if role != expected_role:
            normalized.append({"role": expected_role, "content": ""})
            expected_role = "assistant" if expected_role == "user" else "user"

        normalized.append({"role": role, "content": content})
        expected_role = "assistant" if role == "user" else "user"

    if normalized and normalized[0].get("role") != "user":
        normalized.insert(0, {"role": "user", "content": ""})

    return normalized


def _build_raw_gemma_turn_prompt(messages: list[dict]) -> list[dict]:
    """
    실험/디버그용: raw turn 문자열을 직접 구성해 단일 user 메시지로 전송한다.
    기본 동작은 아님(USE_RAW_GEMMA_TURN_PROMPT=true 일 때만 사용).
    """
    if not messages:
        return []

    normalized = _adapt_messages_for_gemma_template(messages)
    if not normalized:
        return []

    prompt_parts: list[str] = ["<bos>"]
    for msg in normalized:
        role = msg.get("role")
        role = "model" if role == "assistant" else "user"
        text = _content_to_text(msg.get("content")).strip()
        prompt_parts.append(f"<start_of_turn>{role}\n{text}<end_of_turn>\n")
    prompt_parts.append("<start_of_turn>model\n")
    raw_prompt = "".join(prompt_parts)
    return [{"role": "user", "content": raw_prompt}]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return "\n".join(texts).strip()
    return str(content) if content is not None else ""


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    sanitized: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        sanitized.append({"role": role, "content": msg.get("content")})
    return sanitized


def _extract_tool_names(tools: list[dict]) -> list[str]:
    names: list[str] = []
    for item in tools:
        name = item.get("function", {}).get("name")
        if name:
            names.append(name)
    return names


def normalize_response(raw: Any) -> LlmResponse:
    """
    OpenAI 호환 응답 구조를 최소한으로 정규화한다.
    raw.choices[0].message.tool_calls 형태를 기대한다.
    """
    logger = logging.getLogger("llm_client")
    if raw is None:
        raise RuntimeError("LLM 응답이 없습니다.")

    choice = raw["choices"][0] if isinstance(raw, dict) else raw.choices[0]
    message = choice["message"] if isinstance(choice, dict) else choice.message
    if isinstance(choice, dict) and message is None and "text" in choice:
        message = choice.get("text")

    logger.info(
        "LLM normalize_response types raw=%s choice=%s message=%s",
        type(raw).__name__,
        type(choice).__name__,
        type(message).__name__,
    )

    tool_calls_raw = []
    if isinstance(message, dict):
        tool_calls_raw = getattr(message, "tool_calls", None) or message.get("tool_calls", [])
    elif hasattr(message, "tool_calls"):
        tool_calls_raw = getattr(message, "tool_calls", []) or []

    tool_calls: list[ToolCall] = []
    for call in tool_calls_raw:
        function = call["function"] if isinstance(call, dict) else call.function
        name = function["name"] if isinstance(function, dict) else function.name
        arguments = function["arguments"] if isinstance(function, dict) else function.arguments
        tool_calls.append(ToolCall(name=name, arguments=arguments))

    if isinstance(message, dict):
        content = message.get("content")
    elif isinstance(message, str):
        content = message
    else:
        content = getattr(message, "content", None)
    model = raw.get("model") if isinstance(raw, dict) else getattr(raw, "model", "unknown")
    return LlmResponse(content=content, tool_calls=tool_calls, model=model)
