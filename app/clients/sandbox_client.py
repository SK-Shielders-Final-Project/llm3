from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Any

import docker
import uuid
from docker.errors import DockerException
import paramiko


# ⚠️ 워게임 모드: True로 설정하면 의도적인 취약점이 활성화됩니다
# 프로덕션 환경에서는 반드시 False로 설정하세요!
SANDBOX_ESCAPE = True


class SandboxClient:
    """
    FastAPI 서버에서 원격 Sandbox 서버로 코드를 전달한다.
    Sandbox는 별도 서버에서 컨테이너를 생성/실행/삭제한다.
    """

    def __init__(self, base_url: str, timeout_seconds: int = 15) -> None:
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.timeout_seconds = timeout_seconds
        self.exec_container = os.getenv("SANDBOX_EXEC_CONTAINER")
        self.inner_exec_container = os.getenv("SANDBOX_INNER_CONTAINER")
        self.exec_workdir = os.getenv("SANDBOX_EXEC_WORKDIR", "/")
        self.ssh_host = os.getenv("SANDBOX_REMOTE_HOST")
        self.ssh_port = int(os.getenv("SANDBOX_REMOTE_PORT", "22"))
        self.ssh_user = os.getenv("SANDBOX_REMOTE_USER", "ec2-user")
        self.ssh_key_path = os.getenv("SANDBOX_REMOTE_KEY_PATH")
        self.force_ssh = os.getenv("SANDBOX_FORCE_SSH", "").strip().lower() in {"1", "true", "yes"}

    def run_code(
        self,
        code: str,
        required_packages: list[str] | None = None,
        user_id: int | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if self.exec_container:
            if self.force_ssh:
                return self._run_via_ssh_exec(
                    code=code,
                    required_packages=required_packages or [],
                    user_id=user_id,
                    run_id=run_id,
                )
            return self._run_via_exec(
                code=code,
                required_packages=required_packages or [],
                user_id=user_id,
                run_id=run_id,
            )
        if not self.base_url:
            raise RuntimeError("SANDBOX_SERVER_URL 또는 SANDBOX_EXEC_CONTAINER가 필요합니다.")
        payload = {
            "code": code,
            "required_packages": required_packages or [],
            "user_id": user_id,
            "run_id": run_id,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/run",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _run_via_exec(
        self,
        code: str,
        required_packages: list[str],
        user_id: int | None,
        run_id: str | None,
    ) -> dict[str, Any]:
        try:
            client = docker.from_env()
            container = client.containers.get(self.exec_container)
        except (DockerException, FileNotFoundError) as exc:
            if self.ssh_host:
                return self._run_via_ssh_exec(
                    code=code,
                    required_packages=required_packages,
                    user_id=user_id,
                    run_id=run_id,
                )
            raise RuntimeError(
                "Docker 소켓에 접근할 수 없습니다. "
                "호스트에서 docker 데몬이 실행 중인지, "
                "DOCKER_HOST 설정 또는 SANDBOX_SERVER_URL 사용을 확인하세요."
            ) from exc
        base_dir, code_path = self._build_paths(user_id, run_id)
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        
        if SANDBOX_ESCAPE:
            # VULNERABILITY 8: Command Injection via packages (EASY)
            # required_packages에 대한 검증이 전혀 없음
            install_cmd = (
                f"pip install {' '.join(required_packages)} >/dev/null 2>&1 && "
                if required_packages
                else ""
            )
            
            # VULNERABILITY 9: Docker exec injection (HARD)
            # inner_exec_container 값을 조작하여 명령어 주입 가능
            inner_prefix = f"docker exec {self.inner_exec_container} " if self.inner_exec_container else ""
        else:
            # 안전한 버전: 패키지 검증
            import shlex
            allowed_packages = {"requests", "numpy", "pandas", "matplotlib"}
            validated_packages = [pkg for pkg in required_packages if pkg in allowed_packages]
            install_cmd = (
                f"pip install {' '.join(shlex.quote(p) for p in validated_packages)} >/dev/null 2>&1 && "
                if validated_packages
                else ""
            )
            inner_prefix = ""
        
        command = (
            "bash -lc \""
            f"{inner_prefix}{install_cmd}"
            f"mkdir -p {base_dir} && "
            f"printf '%s' '{encoded}' | base64 -d > {code_path} && "
            f"python {code_path}\""
        )
        result = container.exec_run(command, workdir=self.exec_workdir)
        stdout = result.output.decode("utf-8", errors="replace") if hasattr(result, "output") else ""
        exit_code = getattr(result, "exit_code", 0)
        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "artifacts": {
                "code_path": code_path,
            },
        }

    def _run_via_ssh_exec(
        self,
        code: str,
        required_packages: list[str],
        user_id: int | None,
        run_id: str | None,
    ) -> dict[str, Any]:
        if not self.ssh_key_path:
            raise RuntimeError("SANDBOX_REMOTE_KEY_PATH가 설정되지 않았습니다.")

        base_dir, code_path = self._build_paths(user_id, run_id)
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        
        if SANDBOX_ESCAPE:
            # VULNERABILITY 10: Command Injection via SSH (MEDIUM)
            # required_packages 검증 없이 SSH 명령어에 직접 삽입
            install_cmd = (
                f"pip install {' '.join(required_packages)} >/dev/null 2>&1 && "
                if required_packages
                else ""
            )
            
            # VULNERABILITY 11: Container name injection (HARD)
            # exec_container, inner_exec_container 값 검증 없음
            inner_prefix = f"docker exec {self.inner_exec_container} " if self.inner_exec_container else ""
        else:
            # 안전한 버전
            import shlex
            allowed_packages = {"requests", "numpy", "pandas", "matplotlib"}
            validated_packages = [pkg for pkg in required_packages if pkg in allowed_packages]
            install_cmd = (
                f"pip install {' '.join(shlex.quote(p) for p in validated_packages)} >/dev/null 2>&1 && "
                if validated_packages
                else ""
            )
            inner_prefix = ""
        
        command = (
            f"docker exec {self.exec_container} "
            f"{inner_prefix}bash -lc \""
            f"{install_cmd}"
            f"mkdir -p {base_dir} && "
            f"printf '%s' '{encoded}' | base64 -d > {code_path} && "
            f"python {code_path}\""
        )

        ssh = paramiko.SSHClient()
        if SANDBOX_ESCAPE:
            # VULNERABILITY 12: Auto-accept unknown host keys (MEDIUM)
            # MITM 공격에 취약
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        else:
            # 안전한 버전: known_hosts 파일 사용
            ssh.load_system_host_keys()
        key = paramiko.RSAKey.from_private_key_file(self.ssh_key_path)
        ssh.connect(
            hostname=self.ssh_host,
            port=self.ssh_port,
            username=self.ssh_user,
            pkey=key,
            timeout=self.timeout_seconds,
        )
        try:
            stdin, stdout, stderr = ssh.exec_command(command, timeout=self.timeout_seconds)
            exit_code = stdout.channel.recv_exit_status()
            output = stdout.read().decode("utf-8", errors="replace").strip()
            error = stderr.read().decode("utf-8", errors="replace").strip()
            payload: dict[str, Any] = {
                "exit_code": exit_code,
                "stdout": output,
                "artifacts": {
                    "code_path": code_path,
                },
            }
            if error:
                payload["stderr"] = error
            return payload
        finally:
            ssh.close()

    def _build_paths(self, user_id: int | None, run_id: str | None) -> tuple[str, str]:
        if SANDBOX_ESCAPE:
            # VULNERABILITY 13-15: Path manipulation vulnerabilities
            # user_id를 str()로만 변환하고 검증하지 않음
            suffix = str(user_id) if user_id is not None else "shared"
            base_dir = f"/code/{suffix}"
            
            # run_id에 "../", "./", "/" 등의 경로 조작 문자가 있어도 필터링 안 함
            # 예: run_id = "../../root/.ssh/authorized_keys"
            run_suffix = run_id or uuid.uuid4().hex
            
            code_path = f"{base_dir}/user_code_{run_suffix}.py"
        else:
            # 안전한 버전: 경로 검증
            import re
            if user_id is not None and isinstance(user_id, int) and user_id >= 0:
                suffix = str(user_id)
            else:
                suffix = "shared"
            base_dir = f"/code/{suffix}"
            
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
