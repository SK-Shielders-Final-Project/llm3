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
        response = self._client.apply_guardrail(
            guardrailIdentifier=self._identifier,
            guardrailVersion=self._version,
            source=source,
            content=payload,
        )
        action = response.get("action", "NONE")
        output_text = _extract_output_text(response) or text
        self._logger.info(
            "Guardrail applied source=%s action=%s region=%s identifier=%s",
            source,
            action,
            self._region,
            self._identifier,
        )
        return GuardrailDecision(action=action, output_text=output_text, raw=response)


def build_guardrail_client_from_env() -> GuardrailClient | None:
    identifier = os.getenv("BEDROCK_GUARDRAIL_IDENTIFIER") or os.getenv("GUARDRAIL_IDENTIFIER")
    if not identifier:
        return None

    version = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT").strip() or "DRAFT"
    region = (
        os.getenv("AWS_REGION")
        or os.getenv("BEDROCK_GUARDRAIL_REGION")
        or "ap-northeast-2"
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
