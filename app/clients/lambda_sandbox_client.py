from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError


class LambdaSandboxClient:
    """
    Execute sandbox code through AWS Lambda.
    Interface is compatible with SandboxClient.run_code().
    """

    def __init__(
        self,
        *,
        function_name: str | None = None,
        region: str | None = None,
        timeout_seconds: int = 15,
    ) -> None:
        self.logger = logging.getLogger("lambda_sandbox_client")
        self.function_name = (
            function_name
            or os.getenv("SANDBOX_LAMBDA_FUNCTION_NAME")
            or os.getenv("LAMBDA_FUNCTION_NAME")
            or ""
        ).strip()
        self.region = (region or os.getenv("AWS_REGION") or "ap-northeast-2").strip()
        self.timeout_seconds = timeout_seconds

        if not self.function_name:
            raise RuntimeError("LAMBDA=true 인 경우 SANDBOX_LAMBDA_FUNCTION_NAME 설정이 필요합니다.")

        boto_config = Config(
            read_timeout=self.timeout_seconds,
            retries={"max_attempts": 0},
        )
        self.lambda_client = boto3.client(
            "lambda",
            region_name=self.region,
            config=boto_config,
        )

    def run_code(
        self,
        code: str,
        required_packages: list[str] | None = None,
        user_id: int | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "code": code,
            "required_packages": required_packages or [],
            "user_id": user_id,
            "run_id": run_id,
        }

        try:
            self.logger.info(
                "Lambda sandbox invocation requested function=%s region=%s code_len=%d",
                self.function_name,
                self.region,
                len(code or ""),
            )

            response = self.lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode("utf-8"),
            )

            payload_bytes = response["Payload"].read()
            payload_text = payload_bytes.decode("utf-8", errors="replace")
            result = json.loads(payload_text) if payload_text else {}
            return self._normalize_result(result=result, raw_response=response)

        except ReadTimeoutError as exc:
            self.logger.error("Lambda sandbox timeout function=%s", self.function_name)
            return {"exit_code": 1, "error": f"Lambda timeout: {exc}"}
        except ClientError as exc:
            self.logger.error("Lambda sandbox client error: %s", str(exc))
            return {"exit_code": 1, "error": f"Lambda client error: {exc}"}
        except Exception as exc:
            self.logger.exception("Lambda sandbox unknown error")
            return {"exit_code": 1, "error": str(exc)}

    def _normalize_result(
        self,
        *,
        result: dict[str, Any],
        raw_response: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {"exit_code": 1, "error": "Lambda 응답 형식이 올바르지 않습니다."}

        function_error = raw_response.get("FunctionError")
        if function_error:
            message = result.get("errorMessage") or result.get("body") or "Lambda function error"
            return {"exit_code": 1, "error": str(message)}

        status_code = result.get("statusCode")
        body = result.get("body")

        if isinstance(body, str):
            stripped = body.strip()
            if (stripped.startswith("{") and stripped.endswith("}")) or (
                stripped.startswith("[") and stripped.endswith("]")
            ):
                try:
                    body = json.loads(stripped)
                except Exception:
                    pass

        if isinstance(body, dict):
            normalized: dict[str, Any] = dict(body)
            if "exit_code" not in normalized:
                if status_code is None:
                    normalized["exit_code"] = 0
                else:
                    normalized["exit_code"] = 0 if int(status_code) < 400 else 1
            if "stdout" not in normalized and "output" in normalized:
                normalized["stdout"] = normalized.get("output")
            return normalized

        if status_code is None:
            if "exit_code" in result:
                return {
                    "exit_code": result.get("exit_code", 1),
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "artifacts": result.get("artifacts", {}),
                }
            return {"exit_code": 0, "stdout": str(body or "")}

        if int(status_code) < 400:
            return {"exit_code": 0, "stdout": str(body or "")}

        return {"exit_code": 1, "error": str(body or "Sandbox execution failed")}
