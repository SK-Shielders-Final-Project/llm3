from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from typing import Any

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

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
    )


configure_logging()


def create_orchestrator() -> Orchestrator:
    llm_completion = build_http_completion_func()
    llm_client = LlmClient(llm_completion)
    sandbox_url = os.getenv("SANDBOX_SERVER_URL", "")
    sandbox_timeout_raw = os.getenv("SANDBOX_TIMEOUT_SECONDS", "60")
    sandbox_timeout = int(sandbox_timeout_raw) if sandbox_timeout_raw.strip() else 60
    sandbox_client = SandboxClient(base_url=sandbox_url, timeout_seconds=sandbox_timeout)
    registry = FunctionRegistry()
    return Orchestrator(llm_client=llm_client, sandbox_client=sandbox_client, registry=registry)


orchestrator = create_orchestrator()


@app.get("/functions")
def list_functions() -> dict[str, list[str]]:
    """LLM이 호출 가능한 함수 목록을 반환합니다."""
    return {"functions": orchestrator.registry.list_functions()}




@app.post("/api/generate")
def generate(request: GenerateRequest) -> PlainTextResponse:
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
    try:
        result = orchestrator.handle_user_request(message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PlainTextResponse(result["text"])
