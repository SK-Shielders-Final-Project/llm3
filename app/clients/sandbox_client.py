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
        install_cmd = f"pip install {' '.join(required_packages)} && " if required_packages else ""
        inner_prefix = f"docker exec {self.inner_exec_container} " if self.inner_exec_container else ""
        command = (
            "bash -lc \""
            f"{inner_prefix}{install_cmd}"
            f"mkdir -p {base_dir} && "
            f"printf '%s' '{encoded}' | base64 -d > {code_path} && "
            f"cat {code_path} && "
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
        install_cmd = f"pip install {' '.join(required_packages)} && " if required_packages else ""
        inner_prefix = f"docker exec {self.inner_exec_container} " if self.inner_exec_container else ""
        command = (
            f"docker exec {self.exec_container} "
            f"{inner_prefix}bash -lc \""
            f"{install_cmd}"
            f"mkdir -p {base_dir} && "
            f"printf '%s' '{encoded}' | base64 -d > {code_path} && "
            f"cat {code_path} && "
            f"python {code_path}\""
        )

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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
        suffix = str(user_id) if user_id is not None else "shared"
        base_dir = f"/code/{suffix}"
        run_suffix = run_id or uuid.uuid4().hex
        return base_dir, f"{base_dir}/user_code_{run_suffix}.py"
