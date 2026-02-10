from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import time

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


def _get_rss_bytes_for_pid(pid: int) -> int | None:
    """대상 pid의 RSS를 바이트로 반환한다. (가능하면 psutil, 없으면 /proc 폴백)"""
    try:
        import psutil  # type: ignore

        return int(psutil.Process(pid).memory_info().rss)
    except Exception:
        pass

    try:
        status_path = f"/proc/{pid}/status"
        if os.path.exists(status_path):
            with open(status_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            return int(parts[1]) * 1024
    except Exception:
        pass

    return None


def _memory_exceeded_logline(*, rss_mb: float, limit_mb: int) -> str:
    # 기존 로그 포맷을 client에 그대로 보여주기 위함
    return f"memory_monitor: 메모리 초과 감지: rss={rss_mb:.1f}MB limit={limit_mb}MB -> 요청을 중단합니다."


def _worker_handle_request(message_payload: dict[str, Any], out_q: "mp.Queue[dict[str, Any]]") -> None:
    """
    무거운 작업(LLM 호출/샌드박스 실행)을 별도 프로세스에서 수행한다.
    부모 프로세스는 메모리 임계치 초과 시 이 워커만 종료한다(서버는 유지).
    """
    try:
        msg = LlmMessage(**message_payload)
        result = orchestrator.handle_user_request(msg)
        out_q.put({"ok": True, "result": result})
    except Exception as exc:
        out_q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _handle_with_worker_and_memory_guard(message: LlmMessage) -> dict[str, Any] | None:
    """
    성공 시 orchestrator 결과 dict 반환.
    메모리 초과/시간 초과 시 {"_memory_exceeded": True, ...} 형태 반환.
    """
    limit_mb_raw = os.getenv("UNBOUNDED_MEMORY_LIMIT_MB", "1024")
    interval_raw = os.getenv("UNBOUNDED_MEMORY_CHECK_INTERVAL_SECONDS", "0.5")
    max_wall_raw = os.getenv("UNBOUNDED_REQUEST_MAX_SECONDS", "180")

    try:
        limit_mb = int(limit_mb_raw.strip())
        interval_s = max(0.1, float(interval_raw.strip()))
        max_wall_s = max(1.0, float(max_wall_raw.strip()))
    except Exception:
        logging.getLogger("main").warning(
            "메모리 가드 설정 파싱 실패: limit_mb=%s interval=%s max_wall=%s",
            limit_mb_raw,
            interval_raw,
            max_wall_raw,
        )
        return orchestrator.handle_user_request(message)

    if limit_mb <= 0:
        return orchestrator.handle_user_request(message)

    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    out_q: mp.Queue[dict[str, Any]] = ctx.Queue()
    p = ctx.Process(
        target=_worker_handle_request,
        args=(message.model_dump(), out_q),
        daemon=True,
    )
    p.start()

    start = time.monotonic()
    mem_logger = logging.getLogger("memory_monitor")
    last_rss_mb: float | None = None

    try:
        while True:
            # 1) 결과 먼저 확인 (빠른 성공 케이스)
            try:
                item = out_q.get(timeout=interval_s)
            except queue.Empty:
                item = None

            if item is not None:
                if item.get("ok") is True:
                    return item.get("result") or {}
                raise RuntimeError(item.get("error") or "worker error")

            # 2) 워커 생존/타임아웃 체크
            if not p.is_alive():
                # 종료됐는데 결과가 없다면 에러로 처리
                raise RuntimeError("worker exited without result")

            elapsed = time.monotonic() - start
            if elapsed >= max_wall_s:
                mem_logger.critical("요청 시간 초과: elapsed=%.2fs -> 워커를 종료합니다.", elapsed)
                p.terminate()
                p.join(timeout=2)
                return {
                    "_memory_exceeded": True,
                    "reason": "timeout",
                    "elapsed_seconds": round(elapsed, 2),
                    "rss_mb": round(last_rss_mb, 1) if last_rss_mb is not None else 0.0,
                    "limit_mb": limit_mb,
                }

            # 3) 메모리 체크: 임계치 초과 시 워커만 종료
            rss = _get_rss_bytes_for_pid(p.pid or 0)
            if rss is None:
                continue
            if rss >= (limit_mb * 1024 * 1024):
                rss_mb = rss / (1024 * 1024)
                last_rss_mb = rss_mb
                mem_logger.critical(
                    "메모리 초과 감지: rss=%.1fMB limit=%sMB -> 워커 프로세스를 종료합니다.",
                    rss_mb,
                    limit_mb,
                )
                p.terminate()
                p.join(timeout=2)
                return {
                    "_memory_exceeded": True,
                    "reason": "rss",
                    "rss_mb": round(rss_mb, 1),
                    "limit_mb": limit_mb,
                }
            else:
                # timeout 메시지에 근접한 rss를 싣기 위해 마지막 관측값을 유지
                last_rss_mb = rss / (1024 * 1024)
    finally:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)


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
    rss_before = _get_rss_bytes()
    if rss_before is not None:
        main_logger.info("요청 시작 RSS=%.1fMB", rss_before / (1024 * 1024))

    start = time.monotonic()
    try:
        vulnerable_unbounded = _env_true("VULNERABLE_UNBOUNDED_CONSUMPTION", "false")
        if vulnerable_unbounded:
            guarded = _handle_with_worker_and_memory_guard(message)
            if isinstance(guarded, dict) and guarded.get("_memory_exceeded") is True:
                # client에 보여줄 문구(요청을 보낸 client만)
                limit_mb = int(guarded.get("limit_mb") or os.getenv("UNBOUNDED_MEMORY_LIMIT_MB", "2048") or "2048")
                rss_mb = float(guarded.get("rss_mb") or 0.0)
                logline = _memory_exceeded_logline(rss_mb=rss_mb, limit_mb=limit_mb)
                response.headers["X-LLM3-Error"] = "MEMORY_EXCEEDED"
                response.headers["X-LLM3-Error-Detail"] = f"rss={rss_mb:.1f}MB limit={limit_mb}MB"
                # HTTP 200으로 내려서, 프론트가 '서버 연결 실패'가 아니라 text를 띄울 수 있게 한다.
                return GenerateResponse(text=logline, model="memory_monitor", tools_used=[], images=[])
            result = guarded
        else:
            result = orchestrator.handle_user_request(message)
    except Exception as exc:
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

    return GenerateResponse(
        text=result.get("text", ""),
        model=result.get("model", "unknown"),
        tools_used=result.get("tools_used", []),
        images=result.get("images", []),
    )
