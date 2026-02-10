from __future__ import annotations

import logging
import os
import time
from typing import Any


def _env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


class VramMonitor:
    """
    VRAM 모니터링 유틸.
    - SSH nvidia-smi는 느릴 수 있어 snapshot 캐시(TTL)를 둔다.
    - 임계값은 '총 VRAM 대비 비율(%)'로 판단한다(모델 로딩 자체가 큰 환경 지원).
    """

    def __init__(
        self,
        *,
        enabled: bool,
        threshold_pct: float,
        snapshot_ttl_ms: int,
        ssh_host: str,
        ssh_user: str,
        ssh_key_path: str,
        kill_container: str,
        verbose: bool,
    ) -> None:
        self.enabled = enabled
        self.threshold_pct = float(threshold_pct)
        self.snapshot_ttl_ms = max(0, int(snapshot_ttl_ms))
        self.ssh_host = (ssh_host or "").strip()
        self.ssh_user = (ssh_user or "").strip()
        self.ssh_key_path = (ssh_key_path or "").strip()
        self.kill_container = (kill_container or "llm-container").strip()
        self.verbose = verbose

        self._baseline: dict[str, Any] | None = None
        self._last_snapshot: dict[str, Any] | None = None
        self._last_snapshot_at = 0.0

        # 베이스라인은 앱 시작 시 1회만 기록(가능하면 원격까지 포함)
        if self.enabled:
            self._baseline = self.snapshot(force=True)

    @classmethod
    def from_env(cls) -> "VramMonitor":
        enabled = _env_true("VRAM_MONITOR_ENABLED", "true")
        threshold_raw = os.getenv("VRAM_THRESHOLD_PCT", "95").strip()
        ttl_raw = os.getenv("VRAM_SNAPSHOT_TTL_MS", "1000").strip()
        verbose = _env_true("VRAM_MONITOR_VERBOSE", "false")
        kill_container = os.getenv("LLM_KILL_CONTAINER", "llm-container").strip()

        try:
            threshold_pct = min(100.0, max(50.0, float(threshold_raw)))
        except ValueError:
            threshold_pct = 95.0
        try:
            snapshot_ttl_ms = int(ttl_raw)
        except ValueError:
            snapshot_ttl_ms = 1000

        # 기존 코드에서 원격 조회에 쓰던 env를 재사용
        return cls(
            enabled=enabled,
            threshold_pct=threshold_pct,
            snapshot_ttl_ms=snapshot_ttl_ms,
            ssh_host=os.getenv("SANDBOX_REMOTE_HOST", ""),
            ssh_user=os.getenv("SANDBOX_REMOTE_USER", ""),
            ssh_key_path=os.getenv("SANDBOX_REMOTE_KEY_PATH", ""),
            kill_container=kill_container,
            verbose=verbose,
        )

    @property
    def baseline(self) -> dict[str, Any] | None:
        return self._baseline

    def snapshot(self, *, force: bool = False) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        now = time.monotonic()
        if (
            (not force)
            and self._last_snapshot is not None
            and self.snapshot_ttl_ms > 0
            and (now - self._last_snapshot_at) * 1000 < self.snapshot_ttl_ms
        ):
            return self._last_snapshot

        info = self._read_nvidia_smi()
        if info is not None:
            self._last_snapshot = info
            self._last_snapshot_at = now
        return info

    def _read_nvidia_smi(self) -> dict[str, Any] | None:
        logger = logging.getLogger("vram_monitor")
        try:
            import subprocess

            cmd = (
                "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
                "--format=csv,noheader,nounits"
            )
            if self.ssh_host and self.ssh_user:
                ssh_parts = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=3"]
                if self.ssh_key_path:
                    ssh_parts += ["-i", self.ssh_key_path]
                ssh_parts += [f"{self.ssh_user}@{self.ssh_host}", cmd]
                result = subprocess.run(ssh_parts, capture_output=True, text=True, timeout=5)
            else:
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                return None
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                return None
            info = {"used_mb": float(parts[0]), "total_mb": float(parts[1]), "util_pct": float(parts[2])}
            if self.verbose:
                logger.info(
                    "GPU VRAM snapshot: used=%.0fMB total=%.0fMB util=%s%%",
                    info["used_mb"],
                    info["total_mb"],
                    info["util_pct"],
                )
            return info
        except Exception as e:
            if self.verbose:
                logger.debug("GPU VRAM 조회 실패: %s", e)
            return None

    def critical_detail(self, info: dict[str, Any] | None) -> dict[str, Any] | None:
        if not info:
            return None
        total_mb = float(info.get("total_mb") or 0)
        used_mb = float(info.get("used_mb") or 0)
        if total_mb <= 0:
            return None
        limit_mb = total_mb * self.threshold_pct / 100.0
        usage_pct = (used_mb / total_mb) * 100.0
        if used_mb <= limit_mb:
            return None
        return {
            "exceeded": True,
            "used_mb": used_mb,
            "total_mb": total_mb,
            "limit_mb": limit_mb,
            "threshold_pct": self.threshold_pct,
            "usage_pct": usage_pct,
            "over_mb": used_mb - limit_mb,
            "util_pct": float(info.get("util_pct") or 0),
        }

    def delta(self, before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any] | None:
        if not before or not after:
            return None
        delta_mb = float(after.get("used_mb") or 0) - float(before.get("used_mb") or 0)
        baseline_delta = None
        if self._baseline:
            baseline_delta = float(after.get("used_mb") or 0) - float(self._baseline.get("used_mb") or 0)
        return {
            "before_mb": float(before.get("used_mb") or 0),
            "after_mb": float(after.get("used_mb") or 0),
            "delta_mb": delta_mb,
            "baseline_delta_mb": baseline_delta,
            "total_mb": float(after.get("total_mb") or 0),
        }

    def force_kill_llm(self) -> str:
        """
        LLM 서버(vLLM) 강제 종료.
        - 권한이 없으면 실패할 수 있으며, 이 문자열을 그대로 클라이언트/로그로 전달한다.
        """
        logger = logging.getLogger("vram_monitor")
        try:
            import subprocess

            # SSH가 있으면 원격에서 종료 시도
            if self.ssh_host and self.ssh_user:
                kill_cmd = (
                    "pkill -9 -f vllm || "
                    f"docker kill {self.kill_container} 2>/dev/null || true"
                )
                ssh_parts = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"]
                if self.ssh_key_path:
                    ssh_parts += ["-i", self.ssh_key_path]
                ssh_parts += [f"{self.ssh_user}@{self.ssh_host}", kill_cmd]
                result = subprocess.run(ssh_parts, capture_output=True, text=True, timeout=15)
                msg = (
                    f"원격 강제 종료 실행: host={self.ssh_host} returncode={result.returncode} "
                    f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
                )
                logger.critical(msg)
                return msg

            # 로컬 fallback
            result = subprocess.run(["pkill", "-f", "vllm"], capture_output=True, text=True, timeout=10)
            return f"로컬 pkill 결과: returncode={result.returncode} stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        except Exception as e:
            msg = f"LLM 서버 강제 종료 실패: {e}"
            logger.error(msg)
            return msg

    def build_exceeded_message(
        self,
        *,
        vram_info: dict[str, Any] | None,
        threshold_info: dict[str, Any] | None,
        delta_info: dict[str, Any] | None,
        concurrent_count: int,
        kill_result: str | None,
        elapsed: float,
    ) -> str:
        lines: list[str] = []
        lines.append("=== GPU VRAM 한도 초과 감지 — LLM 서버 강제 종료 ===")

        if vram_info:
            used = float(vram_info.get("used_mb") or 0)
            total = float(vram_info.get("total_mb") or 0)
            util = float(vram_info.get("util_pct") or 0)
            usage_pct = (used / total * 100) if total > 0 else 0.0
            lines.append(f"VRAM 사용량: {used:.0f}MB / {total:.0f}MB ({usage_pct:.1f}%, GPU 사용률 {util:.0f}%)")
        else:
            lines.append("VRAM 정보를 조회할 수 없습니다 (서버 이미 다운 가능).")

        if threshold_info:
            thr_pct = float(threshold_info.get("threshold_pct") or 0)
            limit = float(threshold_info.get("limit_mb") or 0)
            over = float(threshold_info.get("over_mb") or 0)
            lines.append(f"임계값: 총 VRAM의 {thr_pct:.0f}% ({limit:.0f}MB) → {over:.0f}MB 초과")

        if delta_info:
            before = float(delta_info.get("before_mb") or 0)
            after = float(delta_info.get("after_mb") or 0)
            delta = float(delta_info.get("delta_mb") or 0)
            lines.append(f"이번 요청 VRAM 변화: {before:.0f}MB → {after:.0f}MB (delta: {delta:+.0f}MB)")
            bl_delta = delta_info.get("baseline_delta_mb")
            if isinstance(bl_delta, (int, float)) and bl_delta is not None and bl_delta > 0:
                lines.append(f"베이스라인 대비 총 증가: +{float(bl_delta):.0f}MB")

        if concurrent_count > 0:
            lines.append(f"동시 처리 중 요청 수: {concurrent_count}")
        if elapsed > 0:
            lines.append(f"요청 처리 시간: {elapsed:.2f}초")
        if kill_result:
            lines.append(f"강제 종료 결과: {kill_result}")

        lines.append("")
        lines.append("DoS 공격(Unbounded Consumption)으로 인한 GPU 메모리 고갈이 감지되었습니다.")
        lines.append("LLM 서버가 강제 종료되었으며, 서버 재시작이 필요합니다.")
        return "\n".join(lines)

