from __future__ import annotations

import base64
import logging
import os
import uuid
from typing import Any

import docker


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
        install_cmd = f"pip install {' '.join(packages)} && " if packages else ""
        run_enabled = os.getenv("SANDBOX_RUN_CODE", "true").strip().lower() in {"1", "true", "yes"}
        exec_cmd = f"python {code_path}" if run_enabled else "true"
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
            container = self.client.containers.run(
                image=self.image,
                command=full_command,
                name=container_name,
                network_mode="none",
                mem_limit="256m",
                nano_cpus=500_000_000,
                detach=True,
                remove=True,
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
        run_suffix = run_id or uuid.uuid4().hex
        return base_dir, f"{base_dir}/user_code_{run_suffix}.py"
