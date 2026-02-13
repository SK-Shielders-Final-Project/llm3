from __future__ import annotations

import logging
import os
import threading
import time
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
from app.service.vram_monitor import VramMonitor
from app.schema import GenerateRequest, GenerateResponse, LlmMessage

app = FastAPI(title="LLM Orchestrator API")
app.include_router(registry_router, prefix="/tools")

# ── 동시 요청 카운터 (DoS 모니터링) ─────────────────────────
_concurrent_requests = 0
_concurrent_lock = threading.Lock()
_peak_concurrent = 0  # 최대 동시 요청 수 기록


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

# VRAM 모니터링은 SSH 호출이 느릴 수 있어 내부 캐시(TTL)를 사용한다.
vram_monitor = VramMonitor.from_env()


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
    # 느린 SSH nvidia-smi 호출은 VramMonitor 내부 TTL 캐시를 사용한다.
    return vram_monitor.snapshot()


def _get_vram_threshold_pct() -> float:
    return float(vram_monitor.threshold_pct)


# ── 앱 시작 시 VRAM 베이스라인 기록 ─────────────────────────
_baseline_vram: dict[str, Any] | None = None


def _record_baseline_vram() -> None:
    """호환성 유지용(실제 기록은 VramMonitor가 수행)."""
    global _baseline_vram
    _baseline_vram = vram_monitor.baseline


def _check_vram_critical(vram_info: dict[str, Any] | None) -> dict[str, Any] | None:
    return vram_monitor.critical_detail(vram_info)


def _calc_vram_delta(
    vram_before: dict[str, Any] | None,
    vram_after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return vram_monitor.delta(vram_before, vram_after)


def _force_kill_llm_server() -> str:
    return vram_monitor.force_kill_llm()


def _build_vram_exceeded_message(
    vram_info: dict[str, Any] | None,
    *,
    threshold_info: dict[str, Any] | None = None,
    delta_info: dict[str, Any] | None = None,
    concurrent_count: int = 0,
    kill_result: str | None = None,
    elapsed: float = 0.0,
) -> str:
    return vram_monitor.build_exceeded_message(
        vram_info=vram_info,
        threshold_info=threshold_info,
        delta_info=delta_info,
        concurrent_count=concurrent_count,
        kill_result=kill_result,
        elapsed=elapsed,
    )


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

# 앱 시작 시 VRAM 베이스라인 기록
_record_baseline_vram()


@app.get("/functions")
def list_functions() -> dict[str, list[str]]:
    """LLM이 호출 가능한 함수 목록을 반환합니다."""
    return {"functions": orchestrator.registry.list_functions()}


@app.post("/api/generate")
def generate(request: GenerateRequest, response: Response) -> GenerateResponse:
    """
    Spring WAS에서 들어온 자연어 요청을 LLM으로 전달하고,
    필요한 함수 및 Sandbox 실행을 오케스트레이션한다.

    VRAM 감지 로직:
    - 요청 전: VRAM 스냅샷 기록 (베이스라인과 비교용)
    - 요청 실패 + LLM 서버 다운 → VRAM 고갈로 판단 → 강제 종료 + 상세 로그 클라이언트 전달
    - 요청 성공 후 VRAM이 총량의 N%(VRAM_THRESHOLD_PCT) 초과 → 경고 헤더 추가
    """
    global _concurrent_requests, _peak_concurrent

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

    # ── 동시 요청 카운터 증가 ──
    with _concurrent_lock:
        _concurrent_requests += 1
        current_concurrent = _concurrent_requests
        if current_concurrent > _peak_concurrent:
            _peak_concurrent = current_concurrent
    main_logger.info("동시 요청 수: %d (최대: %d)", current_concurrent, _peak_concurrent)

    # ── 요청 전 GPU VRAM 상태 기록 (delta 계산용) ──
    vram_before = vram_monitor.snapshot()
    vram_after: dict[str, Any] | None = None

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
            vram_after = vram_monitor.snapshot(force=True)
            vram_final = vram_after or vram_before
            threshold_info = _check_vram_critical(vram_final)
            delta_info = _calc_vram_delta(vram_before, vram_after)

            # 강제 종료 실행
            kill_result = _force_kill_llm_server()

            vram_logger.critical(
                "LLM 서버 다운 감지 (VRAM 고갈): elapsed=%.2fs vram=%s delta=%s threshold=%s concurrent=%d error=%s",
                elapsed,
                vram_final,
                delta_info,
                threshold_info,
                current_concurrent,
                error_str[:200],
            )

            vram_msg = _build_vram_exceeded_message(
                vram_final,
                threshold_info=threshold_info,
                delta_info=delta_info,
                concurrent_count=current_concurrent,
                kill_result=kill_result,
                elapsed=elapsed,
            )

            # HTTP 200으로 내려서, 프론트가 '서버 연결 실패'가 아니라 text를 띄울 수 있게 한다.
            response.headers["X-LLM3-Error"] = "VRAM_EXCEEDED"
            if vram_final:
                response.headers["X-LLM3-VRAM-Used-MB"] = str(int(vram_final.get("used_mb", 0)))
                response.headers["X-LLM3-VRAM-Total-MB"] = str(int(vram_final.get("total_mb", 0)))
            response.headers["X-LLM3-VRAM-Threshold-PCT"] = str(int(_get_vram_threshold_pct()))
            response.headers["X-LLM3-Kill-Action"] = "force_terminated"
            return GenerateResponse(
                text=vram_msg,
                model="vram_monitor",
                tools_used=[],
                images=[],
            )

        raise HTTPException(status_code=500, detail=error_detail) from exc
    finally:
        elapsed = time.monotonic() - start
        # ── 동시 요청 카운터 감소 ──
        with _concurrent_lock:
            _concurrent_requests -= 1

        # ── 요청 후 GPU VRAM 상태 로그 ──
        if vram_after is None:
            vram_after = vram_monitor.snapshot()
        if vram_before and vram_after:
            delta = vram_after["used_mb"] - vram_before["used_mb"]
            main_logger.info(
                "요청 종료 elapsed=%.2fs VRAM=%.0fMB/%.0fMB (delta=%+.0fMB) concurrent=%d",
                elapsed,
                vram_after["used_mb"],
                vram_after["total_mb"],
                delta,
                _concurrent_requests,
            )
        else:
            main_logger.info("요청 종료 elapsed=%.2fs", elapsed)

    # ── 요청 성공 후 VRAM 위험 수준 체크 → 경고 헤더 추가 (결과는 정상 반환) ──
    if vram_after:
        post_critical = _check_vram_critical(vram_after)
        if post_critical:
            vram_logger.warning(
                "[POST-CHECK] VRAM 위험 수준: used=%.0fMB/%.0fMB (%.1f%%, 임계=%s%%)",
                post_critical["used_mb"],
                post_critical["total_mb"],
                post_critical["usage_pct"],
                post_critical["threshold_pct"],
            )
            response.headers["X-LLM3-Warning"] = "VRAM_THRESHOLD_EXCEEDED"
            response.headers["X-LLM3-VRAM-Used-MB"] = str(int(post_critical["used_mb"]))
            response.headers["X-LLM3-VRAM-Total-MB"] = str(int(post_critical["total_mb"]))
            response.headers["X-LLM3-VRAM-Usage-PCT"] = f"{post_critical['usage_pct']:.1f}"

    return GenerateResponse(
        text=result.get("text", ""),
        model=result.get("model", "unknown"),
        tools_used=result.get("tools_used", []),
        images=result.get("images", []),
    )


# ── VRAM 모니터링 엔드포인트 ─────────────────────────────────

@app.get("/api/vram")
def get_vram_status() -> dict[str, Any]:
    """
    GPU VRAM 상태, 임계값 초과 여부, 동시 요청 수, LLM 서버 상태를 반환한다.
    클라이언트/모니터링 도구에서 주기적으로 폴링하여 DoS 상황을 감지할 수 있다.
    """
    vram_info = vram_monitor.snapshot()
    threshold_pct = _get_vram_threshold_pct()
    critical = _check_vram_critical(vram_info)
    llm_alive = _is_llm_server_alive()

    baseline_delta = None
    if vram_monitor.baseline and vram_info:
        baseline_delta = vram_info["used_mb"] - vram_monitor.baseline["used_mb"]

    return {
        "vram": vram_info,
        "baseline_vram": vram_monitor.baseline,
        "baseline_delta_mb": baseline_delta,
        "threshold_pct": threshold_pct,
        "critical": critical is not None,
        "critical_detail": critical,
        "llm_server_alive": llm_alive,
        "concurrent_requests": _concurrent_requests,
        "peak_concurrent_requests": _peak_concurrent,
        "status": "CRITICAL" if (critical or not llm_alive) else "OK",
    }


@app.post("/api/vram/kill")
def force_kill_llm(response: Response) -> dict[str, Any]:
    """
    LLM 서버를 수동으로 강제 종료한다. (관리자/모니터링 용도)
    """
    vram_logger = logging.getLogger("vram_monitor")
    vram_info = vram_monitor.snapshot(force=True)
    kill_result = _force_kill_llm_server()

    vram_logger.critical(
        "[MANUAL KILL] LLM 서버 수동 강제 종료 요청: vram=%s kill_result=%s",
        vram_info,
        kill_result,
    )

    return {
        "action": "force_kill",
        "vram_at_kill": vram_info,
        "kill_result": kill_result,
        "concurrent_requests": _concurrent_requests,
    }
