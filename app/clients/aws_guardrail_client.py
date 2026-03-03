from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import boto3


@dataclass
class GuardrailDecision:
    action: str
    output_text: str | None
    raw: dict[str, Any]


class GuardrailClientProtocol(Protocol):
    def apply(self, *, text: str, source: str) -> GuardrailDecision: ...


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
                "AWS Guardrail apply failed source=%s region=%s identifier=%s",
                source,
                self._region,
                self._identifier,
            )
            raise
        action = response.get("action", "NONE")
        output_text = _extract_output_text(response) or text
        changed = (output_text or "").strip() != text.strip()
        self._logger.info(
            "AWS Guardrail applied source=%s action=%s changed=%s input_len=%d output_len=%d region=%s identifier=%s",
            source,
            action,
            changed,
            len(text),
            len(output_text or ""),
            self._region,
            self._identifier,
        )
        return GuardrailDecision(action=action, output_text=output_text, raw=response)


def _env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _lakera_enabled() -> bool:
    return _env_true("LAKERA_GUARDRAIL_ENABLED", "false")


def _nemo_enabled() -> bool:
    return _env_true("NEMO_GUARDRAIL_ENABLED", "false")


def build_aws_guardrail_client_from_env() -> GuardrailClient | None:
    enabled = os.getenv("BEDROCK_GUARDRAIL_ENABLED", "true").strip().lower()
    if enabled not in ("1", "true", "yes"):
        logging.getLogger("guardrail_client").info("AWS Guardrail disabled via BEDROCK_GUARDRAIL_ENABLED=%s", enabled)
        return None

    identifier = os.getenv("BEDROCK_GUARDRAIL_IDENTIFIER") or os.getenv("GUARDRAIL_IDENTIFIER")
    if not identifier:
        logging.getLogger("guardrail_client").warning(
            "AWS Guardrail enabled but identifier is missing. Set BEDROCK_GUARDRAIL_IDENTIFIER or GUARDRAIL_IDENTIFIER."
        )
        return None

    version = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT").strip() or "DRAFT"
    region = (
        os.getenv("AWS_REGION")
        or os.getenv("BEDROCK_GUARDRAIL_REGION")
        or "ap-northeast-2"
    )
    logging.getLogger("guardrail_client").info(
        "AWS Guardrail client configured enabled=true region=%s version=%s identifier=%s",
        region,
        version,
        identifier,
    )
    return GuardrailClient(identifier=identifier, version=version, region=region)


def build_guardrail_client_from_env() -> GuardrailClientProtocol | None:
    logger = logging.getLogger("guardrail_client")
    print("[DEBUG] build_guardrail_client_from_env() started")
    
    # ── Unified NeMo Guardrails Client ──
    try:
        nemo_enabled = (
            _env_true("NEMO_GUARDRAIL_ENABLED", "false") or 
            _env_true("NEMO_LIBRARY_ENABLED", "false")
        )
        print(f"[DEBUG] NEMO status: {nemo_enabled}")
        
        if nemo_enabled:
            print("[DEBUG] Attempting to build NemoClient...")
            try:
                from app.clients.nemo_client import build_nemo_client
                client = build_nemo_client()
                if client is not None:
                    logger.info("Active guardrail provider=nemo (Unified Client)")
                    print("[DEBUG] NemoClient built successfully")
                    return client
                print("[DEBUG] NemoClient build returned None")
                logger.warning("NeMo guardrail enabled but client init failed; fallback to next provider")
            except Exception as e:
                print(f"[DEBUG] Error importing/building NemoClient: {e}")
                logger.exception(f"Error initializing NemoClient: {e}")
    except Exception as e:
        print(f"[DEBUG] Error checking NEMO status: {e}")

    try:
        lakera_enabled = _lakera_enabled()
        print(f"[DEBUG] Lakera status: {lakera_enabled}")
        if lakera_enabled:
            from app.clients.lakera_guardrail_client import build_lakera_guardrail_client_from_env
            client = build_lakera_guardrail_client_from_env()
            if client is not None:
                logger.info("Active guardrail provider=lakera")
                print("[DEBUG] LakeraClient built successfully")
                return client
            print("[DEBUG] LakeraClient build returned None")
            logger.warning("Lakera guardrail enabled but client init failed; fallback to AWS guardrail")
    except Exception as e:
        print(f"[DEBUG] Error initializing LakeraClient: {e}")

    try:
        print("[DEBUG] Attempting to build AWS Guardrail...")
        client = build_aws_guardrail_client_from_env()
        if client is not None:
            logger.info("Active guardrail provider=aws")
            print("[DEBUG] AWS Guardrail built successfully")
            return client
        print("[DEBUG] AWS Guardrail build returned None")
    except Exception as e:
        print(f"[DEBUG] Error initializing AWS Guardrail: {e}")

    print("[DEBUG] build_guardrail_client_from_env() returning None")
    return None


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
