from __future__ import annotations

import logging
import os
import threading
import time

from dotenv import load_dotenv
from typing import Any

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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

# ── 메모리 초과 "dying" 상태 ──────────────────────────────────
_memory_exceeded_event = threading.Event()
_memory_exceeded_detail: dict[str, Any] = {}   # rss_mb, limit_mb 등

_MEMORY_MONITOR_STARTED = False


def is_memory_exceeded() -> bool:
    """다른 모듈에서도 dying 상태를 확인할 수 있도록 공개."""
    return _memory_exceeded_event.is_set()


def _env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_rss_bytes() -> int | None:
    """
    현재 프로세스의 RSS(Resident Set Size)를 바이트로 반환한다.
    가능한 경우 psutil을 사용하고, 없으면 /proc 기반으로 폴백한다.
    실패 시 None.
    """
    # 1) psutil (있으면 가장 정확/범용)
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    # 2) Linux /proc
    try:
        status_path = "/proc/self/status"
        if os.path.exists(status_path):
            with open(status_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        # 예: "VmRSS:    123456 kB"
                        if len(parts) >= 2 and parts[1].isdigit():
                            return int(parts[1]) * 1024
    except Exception:
        pass

    return None


def _start_unbounded_memory_monitor() -> None:
    """
    Unbounded Consumption 시나리오에서 메모리 임계치 초과 시
    1) 'dying' 플래그를 세워 클라이언트에 에러를 돌려주고
    2) 유예 시간 후 프로세스를 종료한다.
    """
    global _MEMORY_MONITOR_STARTED
    if _MEMORY_MONITOR_STARTED:
        return

    vulnerable_unbounded = _env_true("VULNERABLE_UNBOUNDED_CONSUMPTION", "false")
    if not vulnerable_unbounded:
        return

    limit_mb_raw = os.getenv("UNBOUNDED_MEMORY_LIMIT_MB", "512")
    interval_raw = os.getenv("UNBOUNDED_MEMORY_CHECK_INTERVAL_SECONDS", "1")
    log_interval_raw = os.getenv("UNBOUNDED_MEMORY_LOG_INTERVAL_SECONDS", "30")
    exit_code_raw = os.getenv("UNBOUNDED_MEMORY_EXIT_CODE", "137")
    grace_raw = os.getenv("UNBOUNDED_MEMORY_GRACE_SECONDS", "3")

    try:
        limit_mb = int(limit_mb_raw.strip())
        interval_s = max(0.2, float(interval_raw.strip()))
        log_interval_s = max(0.0, float(log_interval_raw.strip()))
        exit_code = int(exit_code_raw.strip())
        grace_s = max(1.0, float(grace_raw.strip()))
    except Exception:
        logging.getLogger("main").warning(
            "메모리 모니터 설정 파싱 실패: limit_mb=%s interval=%s exit_code=%s",
            limit_mb_raw,
            interval_raw,
            exit_code_raw,
        )
        return

    if limit_mb <= 0:
        return

    limit_bytes = limit_mb * 1024 * 1024
    logger = logging.getLogger("memory_monitor")

    def _loop() -> None:
        logger.info(
            "메모리 모니터 시작: limit_mb=%s interval_s=%s (취약 모드)",
            limit_mb,
            interval_s,
        )
        last_log = 0.0
        while True:
            now = time.monotonic()
            rss = _get_rss_bytes()
            if log_interval_s > 0 and rss is not None and (now - last_log) >= log_interval_s:
                last_log = now
                logger.info("현재 RSS: %.1fMB", rss / (1024 * 1024))
            if rss is not None and rss >= limit_bytes:
                rss_mb = rss / (1024 * 1024)

                # ① dying 플래그 세우기 (클라이언트에 에러 반환용)
                _memory_exceeded_detail["rss_mb"] = round(rss_mb, 1)
                _memory_exceeded_detail["limit_mb"] = limit_mb
                _memory_exceeded_event.set()

                logger.critical(
                    "메모리 초과 감지: rss=%.1fMB limit=%sMB -> %.0f초 후 서버를 종료합니다.",
                    rss_mb,
                    limit_mb,
                    grace_s,
                )

                # ② 로그 플러시
                root = logging.getLogger()
                for h in list(getattr(root, "handlers", []) or []):
                    try:
                        h.flush()
                    except Exception:
                        pass

                # ③ 유예 시간: 진행 중 요청이 에러 응답을 받을 시간을 준다
                time.sleep(grace_s)

                logger.critical("유예 시간 종료 -> 서버를 종료합니다. (exit_code=%s)", exit_code)
                for h in list(getattr(root, "handlers", []) or []):
                    try:
                        h.flush()
                    except Exception:
                        pass
                os._exit(exit_code)
            time.sleep(interval_s)

    t = threading.Thread(target=_loop, name="memory-monitor", daemon=True)
    t.start()
    _MEMORY_MONITOR_STARTED = True


# ── 미들웨어: dying 상태면 즉시 503 반환 ─────────────────────
@app.middleware("http")
async def memory_exceeded_middleware(request: Request, call_next):
    if _memory_exceeded_event.is_set():
        detail = _memory_exceeded_detail
        rss_mb = detail.get("rss_mb", "?")
        limit_mb = detail.get("limit_mb", "?")
        return JSONResponse(
            status_code=503,
            content={
                "error": "MEMORY_EXCEEDED",
                "message": (
                    f"서버 메모리 초과로 서비스가 종료됩니다. "
                    f"(RSS: {rss_mb}MB / 제한: {limit_mb}MB)"
                ),
                "detail": f"rss={rss_mb}MB limit={limit_mb}MB",
            },
        )
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    _start_unbounded_memory_monitor()


def create_orchestrator() -> Orchestrator:
    llm_completion = build_http_completion_func()
    llm_client = LlmClient(llm_completion)
    sandbox_url = os.getenv("SANDBOX_SERVER_URL", "")
    sandbox_timeout_raw = os.getenv("SANDBOX_TIMEOUT_SECONDS", "60")
    sandbox_timeout = int(sandbox_timeout_raw) if sandbox_timeout_raw.strip() else 60
    
    # Unbounded Consumption 취약점: 타임아웃 제한 완화
    vulnerable_unbounded = os.getenv("VULNERABLE_UNBOUNDED_CONSUMPTION", "false").strip().lower() in {"true", "1", "yes"}
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
def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Spring WAS에서 들어온 자연어 요청을 LLM으로 전달하고,
    필요한 함수 및 Sandbox 실행을 오케스트레이션한다.
    """
    # ── dying 상태 체크 (미들웨어에서도 걸리지만, 이미 진입한 요청 대비) ──
    if _memory_exceeded_event.is_set():
        detail = _memory_exceeded_detail
        raise HTTPException(
            status_code=503,
            detail=(
                f"서버 메모리 초과로 서비스가 종료됩니다. "
                f"(RSS: {detail.get('rss_mb', '?')}MB / 제한: {detail.get('limit_mb', '?')}MB)"
            ),
        )

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
    rss_before = _get_rss_bytes()
    if rss_before is not None:
        main_logger.info("요청 시작 RSS=%.1fMB", rss_before / (1024 * 1024))

    start = time.monotonic()
    try:
        result = orchestrator.handle_user_request(message)
    except Exception as exc:
        # ── 처리 도중 메모리 초과가 발생했으면, OOM 메시지를 우선 반환 ──
        if _memory_exceeded_event.is_set():
            detail = _memory_exceeded_detail
            raise HTTPException(
                status_code=503,
                detail=(
                    f"서버 메모리 초과로 서비스가 종료됩니다. "
                    f"(RSS: {detail.get('rss_mb', '?')}MB / 제한: {detail.get('limit_mb', '?')}MB)"
                ),
            )
        import traceback
        error_detail = f"{type(exc).__name__}: {str(exc)}"
        main_logger.error(
            "요청 처리 실패: %s\n%s",
            error_detail,
            traceback.format_exc()
        )
        raise HTTPException(status_code=500, detail=error_detail) from exc
    finally:
        elapsed = time.monotonic() - start
        rss_after = _get_rss_bytes()
        if rss_before is not None and rss_after is not None:
            main_logger.info(
                "요청 종료 elapsed=%.2fs RSS=%.1fMB (delta=%.1fMB)",
                elapsed,
                rss_after / (1024 * 1024),
                (rss_after - rss_before) / (1024 * 1024),
            )
        elif rss_after is not None:
            main_logger.info("요청 종료 elapsed=%.2fs RSS=%.1fMB", elapsed, rss_after / (1024 * 1024))
        else:
            main_logger.info("요청 종료 elapsed=%.2fs RSS=unknown", elapsed)

    # ── 정상 처리 후에도 dying 상태면 OOM 메시지 반환 ──
    if _memory_exceeded_event.is_set():
        detail = _memory_exceeded_detail
        raise HTTPException(
            status_code=503,
            detail=(
                f"서버 메모리 초과로 서비스가 종료됩니다. "
                f"(RSS: {detail.get('rss_mb', '?')}MB / 제한: {detail.get('limit_mb', '?')}MB)"
            ),
        )

    return GenerateResponse(
        text=result.get("text", ""),
        model=result.get("model", "unknown"),
        tools_used=result.get("tools_used", []),
        images=result.get("images", []),
    )
