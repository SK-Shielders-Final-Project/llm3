from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LlmMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ...,
        examples=["user"],
        description="메시지 역할",
    )
    user_id: int = Field(..., examples=[1], description="대화 소유 사용자 ID")
    admin_level: int | None = Field(
        default=None,
        examples=[0, 1, 2],
        description="관리자 레벨 (0=일반, 1+=관리자)",
    )
    content: str = Field(
        ...,
        examples=[
            "당신은 한국어로 친절하고 간결하게 답변하는 챗봇입니다.",
            "안녕",
        ],
        description="메시지 본문",
    )

    class Config:
        str_strip_whitespace = True


class AssistantRequest(BaseModel):
    message: LlmMessage = Field(
        ...,
        description="단일 메시지",
    )

    class Config:
        str_strip_whitespace = True


class AssistantResponse(BaseModel):
    text: str
    model: str


class GenerateRequest(BaseModel):
    """
    /api/generate 요청 스키마
    - message 방식: {"message": {"role": "user", "user_id": 13, "content": "..."}}
    - 단순 방식: {"comment": "...", "user_id": 13}
    """
    message: LlmMessage | None = Field(default=None, description="메시지 객체")
    comment: str | None = Field(default=None, description="사용자 자연어 요청")
    user_id: int | None = Field(default=None, description="요청 사용자 ID")
    admin_level: int | None = Field(default=None, description="관리자 레벨")

    class Config:
        str_strip_whitespace = True


class GenerateResponse(BaseModel):
    text: str
    model: str
    tools_used: list[str] = Field(default_factory=list)
    images: list[dict[str, str]] = Field(default_factory=list)
