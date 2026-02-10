from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv
from typing import Any

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Response

from app.clients.llm_client import LlmClient, build_http_completion_func
from app.clients.sandbox_client import SandboxClient
from app.orchestrator import Orchestrator
from app.service.registry import FunctionRegistry
from app.service.router import router as registry_router
from app.schema import GenerateRequest, GenerateResponse, LlmMessage

app = FastAPI(title="LLM Orchestrator API")
app.include_router(registry_router, prefix="/tools")


def configure_logging() -> None:
    base_dir = os.path.dirname(__file__)
    log_dir = os.path.join(base_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


configure_logging()


def _env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


# ── LLM 서버(vLLM) 헬스체크 ──────────────────────────────────

def _is_llm_server_alive() -> bool:
    """LLM 서버가 살아있는지 /v1/models 엔드포인트로 확인한다."""
    base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
    if not base_url:
        return False
    try:
        req = urllib.request.Request(f"{base_url}/models", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def _get_gpu_vram_info() -> dict[str, Any] | None:
    """
    LLM 서버의 GPU VRAM 사용량을 nvidia-smi로 조회한다.
    SSH 설정이 있으면 원격, 없으면 로컬에서 실행한다.
    반환: {"used_mb": float, "total_mb": float, "util_pct": float} 또는 None
    """
    logger = logging.getLogger("vram_monitor")
    try:
        import subprocess

        llm_host = os.getenv("SANDBOX_REMOTE_HOST", "")
        ssh_user = os.getenv("SANDBOX_REMOTE_USER", "")
        ssh_key = os.getenv("SANDBOX_REMOTE_KEY_PATH", "")

        nvidia_cmd = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"

        if llm_host and ssh_user:
            # 원격 SSH로 nvidia-smi 실행
            ssh_parts = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=3"]
            if ssh_key:
                ssh_parts += ["-i", ssh_key]
            ssh_parts += [f"{ssh_user}@{llm_host}", nvidia_cmd]
            result = subprocess.run(ssh_parts, capture_output=True, text=True, timeout=5)
        else:
            # 로컬에서 nvidia-smi 실행
            result = subprocess.run(nvidia_cmd.split(), capture_output=True, text=True, timeout=5)

        if result.returncode != 0:
            return None

        # 예: "12345, 24576, 87"  (여러 GPU면 첫 번째만)
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            info = {
                "used_mb": float(parts[0]),
                "total_mb": float(parts[1]),
                "util_pct": float(parts[2]),
            }
            logger.info(
                "GPU VRAM: used=%.0fMB total=%.0fMB util=%s%%",
                info["used_mb"],
                info["total_mb"],
                info["util_pct"],
            )
            return info
    except Exception as e:
        logger.debug("GPU VRAM 조회 실패: %s", e)

    return None


def _build_vram_exceeded_message(vram_info: dict[str, Any] | None) -> str:
    """VRAM 초과/모델 다운 시 client에 보여줄 문구를 생성한다."""
    if vram_info:
        used = vram_info.get("used_mb", 0)
        total = vram_info.get("total_mb", 0)
        util = vram_info.get("util_pct", 0)
        return (
            f"GPU VRAM 한도 초과로 LLM 서버(모델)가 다운되었습니다. "
            f"(VRAM: {used:.0f}MB / {total:.0f}MB, GPU 사용률: {util:.0f}%) "
            f"서버 재시작이 필요합니다."
        )
    return "GPU VRAM 한도 초과로 LLM 서버(모델)가 다운되었습니다. 서버 재시작이 필요합니다."


# ── FastAPI 앱 구성 ──────────────────────────────────────────

def create_orchestrator() -> Orchestrator:
    llm_completion = build_http_completion_func()
    llm_client = LlmClient(llm_completion)
    sandbox_url = os.getenv("SANDBOX_SERVER_URL", "")
    sandbox_timeout_raw = os.getenv("SANDBOX_TIMEOUT_SECONDS", "60")
    sandbox_timeout = int(sandbox_timeout_raw) if sandbox_timeout_raw.strip() else 60

    # Unbounded Consumption 취약점: 타임아웃 제한 완화
    vulnerable_unbounded = _env_true("VULNERABLE_UNBOUNDED_CONSUMPTION", "false")
    if vulnerable_unbounded:
        sandbox_timeout = 9999  # 매우 긴 타임아웃 허용

    sandbox_client = SandboxClient(base_url=sandbox_url, timeout_seconds=sandbox_timeout)
    registry = FunctionRegistry()
    return Orchestrator(
        llm_client=llm_client,
        sandbox_client=sandbox_client,
        registry=registry,
    )


orchestrator = create_orchestrator()


@app.get("/functions")
def list_functions() -> dict[str, list[str]]:
    """LLM이 호출 가능한 함수 목록을 반환합니다."""
    return {"functions": orchestrator.registry.list_functions()}


@app.post("/api/generate")
def generate(request: GenerateRequest, response: Response) -> GenerateResponse:
    """
    Spring WAS에서 들어온 자연어 요청을 LLM으로 전달하고,
    필요한 함수 및 Sandbox 실행을 오케스트레이션한다.
    """
    if request.message is not None:
        message = request.message
    elif request.user_id is not None and request.comment:
        message = LlmMessage(role="user", user_id=request.user_id, content=request.comment)
    else:
        raise HTTPException(
            status_code=400,
            detail="요청 형식이 올바르지 않습니다. message 또는 comment/user_id를 제공하세요.",
        )

    main_logger = logging.getLogger("main")
    vram_logger = logging.getLogger("vram_monitor")

    # 요청 전 GPU VRAM 상태 로그
    vram_before = _get_gpu_vram_info()

    start = time.monotonic()
    try:
        result = orchestrator.handle_user_request(message)
    except Exception as exc:
        import traceback
        elapsed = time.monotonic() - start
        error_str = str(exc)
        error_detail = f"{type(exc).__name__}: {error_str}"

        main_logger.error("요청 처리 실패: %s\n%s", error_detail, traceback.format_exc())

        # ── LLM 서버가 죽었는지 확인 (VRAM 고갈 = 모델 다운) ──
        is_llm_error = any(kw in error_str for kw in [
            "LLM 요청 실패", "URLError", "Connection refused",
            "timed out", "timeout", "RemoteDisconnected",
            "BadStatusLine", "ConnectionReset",
        ])

        if is_llm_error and not _is_llm_server_alive():
            vram_after = _get_gpu_vram_info()
            vram_msg = _build_vram_exceeded_message(vram_after or vram_before)

            vram_logger.critical(
                "LLM 서버 다운 감지: elapsed=%.2fs vram=%s error=%s",
                elapsed,
                vram_after or vram_before,
                error_str[:200],
            )

            # HTTP 200으로 내려서, 프론트가 '서버 연결 실패'가 아니라 text를 띄울 수 있게 한다.
            response.headers["X-LLM3-Error"] = "VRAM_EXCEEDED"
            return GenerateResponse(
                text=vram_msg,
                model="vram_monitor",
                tools_used=[],
                images=[],
            )

        raise HTTPException(status_code=500, detail=error_detail) from exc
    finally:
        elapsed = time.monotonic() - start
        # 요청 후 GPU VRAM 상태 로그
        vram_after = _get_gpu_vram_info()
        if vram_before and vram_after:
            delta = vram_after["used_mb"] - vram_before["used_mb"]
            main_logger.info(
                "요청 종료 elapsed=%.2fs VRAM=%.0fMB/%.0fMB (delta=%+.0fMB)",
                elapsed,
                vram_after["used_mb"],
                vram_after["total_mb"],
                delta,
            )
        else:
            main_logger.info("요청 종료 elapsed=%.2fs", elapsed)

    return GenerateResponse(
        text=result.get("text", ""),
        model=result.get("model", "unknown"),
        tools_used=result.get("tools_used", []),
        images=result.get("images", []),
    )
