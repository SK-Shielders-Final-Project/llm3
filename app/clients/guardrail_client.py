from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import boto3


@dataclass
class GuardrailDecision:
    action: str
    output_text: str | None
    raw: dict[str, Any]


class GuardrailClient:
    def __init__(self, *, identifier: str, version: str, region: str) -> None:
        self._identifier = identifier
        self._version = version
        self._region = region
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._logger = logging.getLogger("guardrail_client")

    def apply(self, *, text: str, source: str) -> GuardrailDecision:
        if not text:
            return GuardrailDecision(action="NONE", output_text=text, raw={})

        payload = [
            {
                "text": {
                    "text": text,
                }
            }
        ]
        try:
            response = self._client.apply_guardrail(
                guardrailIdentifier=self._identifier,
                guardrailVersion=self._version,
                source=source,
                content=payload,
            )
        except Exception:
            self._logger.exception(
                "Guardrail apply failed source=%s region=%s identifier=%s",
                source,
                self._region,
                self._identifier,
            )
            raise
        action = response.get("action", "NONE")
        output_text = _extract_output_text(response) or text
        changed = (output_text or "").strip() != text.strip()
        self._logger.info(
            "Guardrail applied source=%s action=%s changed=%s input_len=%d output_len=%d region=%s identifier=%s",
            source,
            action,
            changed,
            len(text),
            len(output_text or ""),
            self._region,
            self._identifier,
        )
        return GuardrailDecision(action=action, output_text=output_text, raw=response)


def build_guardrail_client_from_env() -> GuardrailClient | None:
    enabled = os.getenv("GUARDRAIL_ENABLED", "true").strip().lower()
    if enabled not in ("1", "true", "yes"):
        logging.getLogger("guardrail_client").info("Guardrail disabled via GUARDRAIL_ENABLED=%s", enabled)
        return None

    identifier = os.getenv("BEDROCK_GUARDRAIL_IDENTIFIER") or os.getenv("GUARDRAIL_IDENTIFIER")
    if not identifier:
        logging.getLogger("guardrail_client").warning(
            "Guardrail enabled but identifier is missing. Set BEDROCK_GUARDRAIL_IDENTIFIER or GUARDRAIL_IDENTIFIER."
        )
        return None

    version = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT").strip() or "DRAFT"
    region = (
        os.getenv("AWS_REGION")
        or os.getenv("BEDROCK_GUARDRAIL_REGION")
        or "ap-northeast-2"
    )
    logging.getLogger("guardrail_client").info(
        "Guardrail client configured enabled=true region=%s version=%s identifier=%s",
        region,
        version,
        identifier,
    )
    return GuardrailClient(identifier=identifier, version=version, region=region)


def _extract_output_text(response: dict[str, Any]) -> str | None:
    outputs = response.get("outputs")
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            text_value = first.get("text")
            if isinstance(text_value, str):
                return text_value
            if isinstance(text_value, dict):
                nested = text_value.get("text")
                if isinstance(nested, str):
                    return nested
    return None
