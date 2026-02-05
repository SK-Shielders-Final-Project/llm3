from __future__ import annotations

import base64
import logging
import os
import uuid
from typing import Any

import docker


# ⚠️ 워게임 모드: True로 설정하면 의도적인 취약점이 활성화됩니다
# 프로덕션 환경에서는 반드시 False로 설정하세요!
SANDBOX_ESCAPE = True


class SandboxManager:
    def __init__(self) -> None:
        self.image = "python:3.10-slim"
        self.remote_host = os.getenv("SANDBOX_REMOTE_HOST")
        self.remote_port = int(os.getenv("SANDBOX_REMOTE_PORT", "2375"))
        self.remote_user = os.getenv("SANDBOX_REMOTE_USER", "ec2-user")
        self.remote_key_path = os.getenv("SANDBOX_REMOTE_KEY_PATH")
        self.client = self._build_client()

    def run_code(
        self,
        code: str,
        packages: list[str] | None = None,
        timeout: int = 15,
        user_id: int | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        container_name = f"sandbox_{uuid.uuid4().hex}"
        packages = packages or []
        logger = logging.getLogger("sandbox")

        base_dir, code_path = self._build_paths(user_id, run_id)
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        
        # VULNERABILITY 1: Command Injection via packages (EASY)
        # 패키지 이름에 대한 검증이 없어서 명령어 주입 가능
        # 예: packages = ["requests; cat /etc/passwd"]
        if SANDBOX_ESCAPE:
            # 취약한 버전: 검증 없이 그대로 사용
            install_cmd = f"pip install {' '.join(packages)} && " if packages else ""
        else:
            # 안전한 버전: 패키지 검증
            import shlex
            allowed_packages = {"requests", "numpy", "pandas", "matplotlib"}
            validated_packages = [pkg for pkg in packages if pkg in allowed_packages]
            install_cmd = f"pip install {' '.join(shlex.quote(p) for p in validated_packages)} && " if validated_packages else ""
        
        run_enabled = os.getenv("SANDBOX_RUN_CODE", "true").strip().lower() in {"1", "true", "yes"}
        exec_cmd = f"python {code_path}" if run_enabled else "true"
        
        # VULNERABILITY 2: Path Traversal via run_id (MEDIUM)
        # run_id를 통해 경로 조작이 가능하여 컨테이너 외부 파일 접근 가능
        full_command = (
            "sh -c \""
            f"{install_cmd}"
            f"mkdir -p {base_dir} && "
            f"printf '%s' '{encoded}' | base64 -d > {code_path} && "
            f"cat {code_path} && "
            f"{exec_cmd}\""
        )
        logger.info("Sandbox run_enabled=%s command=%s", run_enabled, full_command)

        container = None
        try:
            if SANDBOX_ESCAPE:
                # VULNERABILITY 3: Weak container isolation (HARD)
                # Docker 소켓 마운트로 컨테이너 탈출 가능
                volumes = {"/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}}
                
                # VULNERABILITY 4: Privileged mode option (CRITICAL)
                privileged = True
                
                # VULNERABILITY 5: Network not isolated (MEDIUM)
                # 네트워크 접근 허용
                network_mode = "bridge"
                
                container = self.client.containers.run(
                    image=self.image,
                    command=full_command,
                    name=container_name,
                    network_mode=network_mode,
                    mem_limit="256m",
                    nano_cpus=500_000_000,
                    detach=True,
                    remove=True,
                    privileged=privileged,
                    volumes=volumes,
                )
            else:
                # 안전한 버전: 격리된 컨테이너
                container = self.client.containers.run(
                    image=self.image,
                    command=full_command,
                    name=container_name,
                    network_mode="none",
                    mem_limit="256m",
                    nano_cpus=500_000_000,
                    detach=True,
                    remove=True,
                    privileged=False,
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges"],
                )
            result = container.wait(timeout=timeout)
            logs = container.logs().decode("utf-8", errors="replace")
            return {
                "exit_code": result.get("StatusCode"),
                "stdout": logs,
                "artifacts": {
                    "code_path": code_path,
                },
            }
        except Exception as exc:
            return {"exit_code": 1, "error": str(exc)}
        finally:
            if container is not None:
                try:
                    container.remove(force=False)
                except Exception:
                    pass

    def _build_client(self) -> docker.DockerClient:
        if self.remote_host:
            return docker.DockerClient(base_url=f"tcp://{self.remote_host}:{self.remote_port}")
        return docker.from_env()

    def _build_paths(self, user_id: int | None, run_id: str | None) -> tuple[str, str]:
        suffix = str(user_id) if user_id is not None else "shared"
        base_dir = f"/code/{suffix}"
        
        if SANDBOX_ESCAPE:
            # VULNERABILITY 6-7: Path Traversal & Arbitrary file write (EASY-MEDIUM)
            # run_id에 "../" 같은 경로 조작 문자열이 포함되어도 필터링하지 않음
            # 예: run_id = "../../etc/passwd"
            run_suffix = run_id or uuid.uuid4().hex
            code_path = f"{base_dir}/user_code_{run_suffix}.py"
        else:
            # 안전한 버전: 경로 검증
            import re
            if run_id:
                # 영문, 숫자, 하이픈, 언더스코어만 허용
                if not re.match(r'^[a-zA-Z0-9_-]+$', run_id):
                    run_id = None
                run_suffix = run_id or uuid.uuid4().hex
            else:
                run_suffix = uuid.uuid4().hex
            
            code_path = os.path.normpath(f"{base_dir}/user_code_{run_suffix}.py")
            # 경로가 base_dir 내부인지 확인
            if not code_path.startswith(base_dir):
                code_path = f"{base_dir}/user_code_{uuid.uuid4().hex}.py"
        
        return base_dir, code_path
