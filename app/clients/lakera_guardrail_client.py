from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from app.clients.aws_guardrail_client import GuardrailDecision


class LakeraGuardrailClient:
    def __init__(self, *, api_key: str, endpoint: str, project_id: str = "", timeout_seconds: int = 10) -> None:
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._project_id = project_id
        self._timeout_seconds = timeout_seconds
        self._logger = logging.getLogger("guardrail_client")

    def apply(self, *, text: str, source: str) -> GuardrailDecision:
        if not text:
            return GuardrailDecision(action="NONE", output_text=text, raw={})

        payload: dict[str, Any] = {
            "input": text,
            "source": source,
        }
        if self._project_id:
            payload["project_id"] = self._project_id
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
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
            self._logger.error("Lakera Guardrail HTTPError status=%s detail=%s", exc.code, detail)
            raise
        except Exception:
            self._logger.exception("Lakera Guardrail apply failed endpoint=%s", self._endpoint)
            raise

        action, output_text = _extract_lakera_result(raw_response, original_text=text)
        changed = (output_text or "").strip() != text.strip()
        self._logger.info(
            "Lakera Guardrail applied source=%s action=%s changed=%s input_len=%d output_len=%d endpoint=%s",
            source,
            action,
            changed,
            len(text),
            len(output_text or ""),
            self._endpoint,
        )
        return GuardrailDecision(action=action, output_text=output_text, raw=raw_response)


def build_lakera_guardrail_client_from_env() -> LakeraGuardrailClient | None:
    enabled = (
        os.getenv("LAKERA_GUARDRAIL_ENABLED", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    )
    if not enabled:
        logging.getLogger("guardrail_client").info("Lakera Guardrail disabled")
        return None

    api_key = (os.getenv("LAKERA_API_KEY") or "").strip()
    if not api_key:
        logging.getLogger("guardrail_client").warning("Lakera Guardrail enabled but LAKERA_API_KEY is missing")
        return None

    project_id = (os.getenv("LAKERA_PROJECT_ID") or "").strip()
    endpoint = (os.getenv("LAKERA_GUARDRAIL_URL") or "https://api.lakera.ai/v2/guard").strip()
    timeout_raw = (os.getenv("LAKERA_TIMEOUT_SECONDS") or "10").strip()
    try:
        timeout_seconds = int(timeout_raw)
    except ValueError:
        timeout_seconds = 10
    logging.getLogger("guardrail_client").info(
        "Lakera Guardrail configured enabled=true endpoint=%s project_id=%s timeout=%s",
        endpoint,
        project_id,
        timeout_seconds,
    )
    return LakeraGuardrailClient(
        api_key=api_key,
        endpoint=endpoint,
        project_id=project_id,
        timeout_seconds=timeout_seconds,
    )


def _extract_lakera_result(response: dict[str, Any], *, original_text: str) -> tuple[str, str]:
    if not isinstance(response, dict):
        return "NONE", original_text

    action = str(response.get("action") or "NONE").upper()
    blocked = bool(response.get("blocked") or response.get("flagged") or response.get("is_blocked"))
    output_text = response.get("output_text") or response.get("text") or original_text

    if isinstance(output_text, dict):
        output_text = output_text.get("text") or original_text
    if not isinstance(output_text, str):
        output_text = original_text

    if blocked and action == "NONE":
        action = "BLOCK"
    if blocked:
        output_text = ""

    return action, output_text
