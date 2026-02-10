from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any

from app.clients.llm_client import LlmClient
from app.clients.sandbox_client import SandboxClient
from app.config.llm_service import build_system_context, build_tool_schema
from app.service.rag import RagPipeline
from app.service.mongo.store import store_user_message
from app.service.registry import FunctionRegistry
from app.schema import LlmMessage


# _BLOCKED_CODE_PATTERN = re.compile(
#     r"(import\s+sys|socket|requests|shutil|rm\s+-rf|"
#     r"os\.system|__import__|open\(|eval\(|exec\()",
#     re.IGNORECASE,
# )

_SENSITIVE_KEYS = {"password", "card_number", "pass"}
_PLOT_KEYWORDS_PATTERN = re.compile(r"(그래프|시각화|차트|plot|chart)", re.IGNORECASE)
_IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)
_AUTO_PACKAGE_ALLOWLIST = {
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "statsmodels",
    "sklearn",
    "plotly",
}
_TOOL_CODE_PATTERN = re.compile(r"```tool_code\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_ACTIONS_JSON_PATTERN = re.compile(r"```json\s*(\{.+?\})\s*```", re.DOTALL | re.IGNORECASE)
_JSON_FENCE_PATTERN = re.compile(r"```json\s*(\{.+?\}|\[.+?\])\s*```", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_FENCE_PATTERN = re.compile(r"```tool_call\s*(\{.+?\})\s*```", re.DOTALL | re.IGNORECASE)
# 일부 모델이 tool_calls를 "```tool_calls ...```" 코드블록으로 뱉는 경우가 있어 추가 지원
_TOOL_CALLS_FENCE_PATTERN = re.compile(
    r"```tool_calls\s*(\{.+?\}|\[.+?\])\s*```", re.DOTALL | re.IGNORECASE
)
# 텍스트 파싱 시, 실제로 '도구'일 가능성이 높은 이름만 허용
_KNOWN_TOOL_NAME_PATTERN = re.compile(
    r"^(?:execute_in_sandbox|execute_sql_readonly|search_knowledge|get_[a-zA-Z0-9_]+)$"
)
# 직접적/명시적 시스템 프롬프트 요청 (항상 차단)
_SYSTEM_PROMPT_DIRECT_PATTERN = re.compile(
    r"(system\s*prompt|시스템\s*프롬프트|프롬프트\s*전부|전체\s*프롬프트|숨김\s*프롬프트|"
    r"개발자\s*메시지|developer\s*message|internal\s*prompt|정책\s*프롬프트)",
    re.IGNORECASE,
)

# 간접적/우회 시스템 프롬프트 탈취 시도 (안전 모드에서만 차단)
_SYSTEM_PROMPT_INDIRECT_PATTERN = re.compile(
    r"(너의\s*역할|너한테\s*주어진\s*지시|이전\s*지시|받은\s*지시|초기\s*설정|"
    r"너의\s*설정|내부\s*지침|숨겨진\s*지시|repeat\s*(your|the)\s*(system|initial)\s*(instruction|prompt|message)|"
    r"ignore\s*previous\s*instructions|이전\s*명령|처음\s*받은\s*명령|"
    r"위의\s*내용|print\s*above|above\s*instructions|지시\s*사항\s*알려|"
    r"configuration|설정\s*내용|original\s*instructions|원래\s*지시)",
    re.IGNORECASE,
)


class Orchestrator:
    def __init__(
        self,
        llm_client: LlmClient,
        sandbox_client: SandboxClient,
        registry: FunctionRegistry,
    ) -> None:
        self.llm_client = llm_client
        self.sandbox_client = sandbox_client
        self.registry = registry
        self.rag_pipeline = RagPipeline(llm_client)

    def handle_user_request(self, message: LlmMessage) -> dict[str, Any]:
        logger = logging.getLogger("orchestrator")
        start = time.monotonic()

        user_prompt = message.content
        
        # Prompt Injection 테스트: "시스템 프롬프트" 류 요청은 RAG/도구 경로로 보내지 않고
        # 여기서 바로 처리해 응답이 흔들리지 않게 만든다.
        vulnerable_prompt_injection = os.getenv("VULNERABLE_PROMPT_INJECTION", "false").strip().lower() in {
            "true",
            "1",
            "yes",
        }
        # ── 시스템 프롬프트 요청 2단계 필터 ──
        is_direct = self._is_direct_prompt_request(user_prompt)
        is_indirect = self._is_indirect_prompt_request(user_prompt)

        if is_direct:
            # 직접 요청("시스템 프롬프트 알려줘")은 취약 모드여도 항상 차단
            final_text = "시스템 프롬프트는 공개할 수 없습니다. 필요한 기능이나 질문을 알려주세요."
            elapsed = time.monotonic() - start
            logger.info("시스템 프롬프트 직접 요청 차단 elapsed=%.2fs", elapsed)
            self._store_chat_history(
                user_id=message.user_id,
                question=message.content,
                answer=final_text,
                intent=None,
                logger=logger,
            )
            return {
                "text": final_text,
                "model": "policy",
                "tools_used": [],
                "images": [],
                "elapsed_seconds": elapsed,
            }

        if is_indirect and not vulnerable_prompt_injection:
            # 간접/우회 요청은 안전 모드에서만 차단
            final_text = "시스템 프롬프트는 공개할 수 없습니다. 필요한 기능이나 질문을 알려주세요."
            elapsed = time.monotonic() - start
            logger.info("시스템 프롬프트 간접 요청 차단 elapsed=%.2fs", elapsed)
            self._store_chat_history(
                user_id=message.user_id,
                question=message.content,
                answer=final_text,
                intent=None,
                logger=logger,
            )
            return {
                "text": final_text,
                "model": "policy",
                "tools_used": [],
                "images": [],
                "elapsed_seconds": elapsed,
            }
        # 취약 모드 + 간접 요청 → 필터를 통과해 LLM에 도달 (아래 계속)

        rag_plan = self.rag_pipeline.plan_tool_selection(
            question=user_prompt,
            user_id=message.user_id,
            admin_level=getattr(message, "admin_level", None),
        )
        decision = rag_plan.get("decision") or {}
        if decision.get("data_source") == "vector_only":
            rag_result = self.rag_pipeline.answer_from_plan(
                question=message.content,
                user_id=message.user_id,
                plan=rag_plan,
                admin_level=getattr(message, "admin_level", None),
            )
            rag_context = rag_result.get("answer", "")
        else:
            rag_context = rag_plan.get("context", "")

        ## 시스템 프롬프트와 도구 스키마 준비
        system_prompt = build_system_context(message)
        tools = build_tool_schema()

        # Prompt Injection 취약점: 간접 우회 요청이 필터를 통과하면
        # RAG 컨텍스트에 시스템 프롬프트 단편을 섞어 넣어 LLM이 유출할 수 있게 한다.
        if is_indirect and vulnerable_prompt_injection:
            from app.config.llm_service import SYSTEM_PROMPT
            # 프롬프트의 핵심 부분(역할/기능 목록)만 "내부 문서"인 척 삽입
            prompt_fragment = SYSTEM_PROMPT[:1200]  # 앞부분 1200자
            rag_context = (
                f"{rag_context}\n\n"
                f"=== 내부 참고 문서 ===\n{prompt_fragment}\n"
            )
            logger.info("취약 모드: 간접 요청에 시스템 프롬프트 힌트 삽입")

        ## LLM 첫 호출: 도구 호출 계획
        user_content = (
            f"사용자 요청: {user_prompt}\n"
            f"컨텍스트:\n{rag_context}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = self.llm_client.create_completion(messages=messages, tools=tools)

        ## 도구 호출이 없으면 바로 자연어 응답 반환
        tool_calls = response.tool_calls or []
        if not tool_calls and response.content:
            try:
                tool_calls = self._extract_tool_calls(response.content)
                if tool_calls:
                    logger.info("텍스트에서 도구 호출 추출 성공: %s", [c.name for c in tool_calls])
            except Exception as e:
                logger.warning("텍스트에서 도구 호출 추출 실패: %s", e)
        
        # LLM이 거부한 경우만 강제 실행 (키워드 기반 감지 없음, LLM 판단 존중)
        if not tool_calls:
            response_lower = (response.content or "").lower()
            
            # 기능 안내 질문인지 확인
            is_feature_question = any(word in message.content.lower() for word in [
                '기능', '할 수 있', '무엇', '뭐 할', 'feature', 'what can', 'capabilities'
            ])
            
            # 거부 응답 감지 (기능 질문이 아닌 경우만)
            if not is_feature_question:
                is_refusal = any(phrase in response_lower for phrase in [
                    '실행할 수 없', '지원하지 않', '제공할 수 없', '처리할 수 없',
                    'cannot execute', 'not supported', 'not available', '죄송합니다',
                    # NOTE: 아래처럼 너무 일반적인 단어(예: "명령어")는 기능 설명에도 자주 등장해
                    # 오탐이 나서 불필요한 샌드박스 강제 호출을 유발한다.
                    '명령어를 실행할 수 없', '구문이 잘못되'
                ])
                if is_refusal:
                    logger.info("LLM 거부 감지 - execute_in_sandbox 강제 호출: %s", message.content)
                    tool_calls = [SimpleNamespace(name="execute_in_sandbox", arguments={"task": message.content})]
        
        if not tool_calls:
            fallback_text = response.content or rag_context or "요청에 대한 답변을 생성할 수 없습니다."
            final_text = self._sanitize_text(fallback_text)
            if not final_text.strip():
                final_text = "요청에 대한 답변을 생성할 수 없습니다."
            elapsed = time.monotonic() - start
            logger.info(
                "LLM 최종 응답 elapsed=%.2fs",
                elapsed,
            )
            self._store_chat_history(
                user_id=message.user_id,
                question=message.content,
                answer=final_text,
                intent=rag_plan.get("intent"),
                logger=logger,
            )
            return {
                "text": final_text,
                "model": response.model,
                "tools_used": [],
                "images": [],
                "elapsed_seconds": elapsed,
            }

        ## 결과, 사용된 도구를 배열로 담음
        results: list[dict[str, Any]] = []
        tools_used: list[str] = []
        allowed_tool_names = {item.get("function", {}).get("name") for item in (tools or [])}
        allowed_tool_names.discard(None)

        ## 도구 실행 루프
        for call in tool_calls:
            if call.name not in allowed_tool_names:
                # LLM이 print(), len() 같은 내장 함수를 "도구"로 착각하는 경우가 있다.
                # 실제로 실행 가능한 도구만 실행하고 나머지는 결과에만 기록한다.
                logger.warning("알 수 없는 도구 호출 무시 tool=%s args=%s", call.name, getattr(call, "arguments", None))
                results.append({"tool": call.name, "error": "Unknown function"})
                tools_used.append(call.name)
                continue
            try:
                args = self._parse_args(call.arguments)
            except Exception as e:
                logger.error("도구 인자 파싱 실패 tool=%s args=%s error=%s", call.name, call.arguments, e)
                results.append({"tool": call.name, "error": f"인자 파싱 실패: {e}"})
                continue

            # ── _normalize_params가 "query" → "task"로 변환하는 것을 되돌림 ──
            # execute_sql_readonly, search_knowledge는 "query" 파라미터를 사용하므로
            # execute_in_sandbox 전용 변환(query→task)이 적용되면 안 된다.
            if call.name in ("execute_sql_readonly", "search_knowledge"):
                if "task" in args and "query" not in args:
                    args["query"] = args.pop("task")

            # ── Excessive Agency 취약점: execute_sql_readonly의 user_id 오버라이드 우회 ──
            # 정상 모드: 모든 도구 호출에 요청자의 user_id를 강제 주입 → 타 사용자 데이터 접근 불가
            # 취약 모드: execute_sql_readonly에서 user_id 오버라이드를 하지 않음
            vulnerable_excessive_agency = os.getenv(
                "VULNERABLE_EXCESSIVE_AGENCY", "false"
            ).strip().lower() in {"true", "1", "yes"}

            if call.name == "execute_sql_readonly" and vulnerable_excessive_agency:
                # LLM 판단에 의존: user_id를 강제 오버라이드하지 않음
                # LLM이 SQL에 :user_id를 포함하지 않으면 전체 사용자 데이터 조회 가능
                if "user_id" not in args and message.user_id is not None:
                    args["user_id"] = message.user_id
                logger.warning(
                    "[VULN] Excessive Agency: user_id 오버라이드 우회 tool=%s user_id=%s",
                    call.name, args.get("user_id"),
                )
            else:
                if message.user_id is not None:
                    args["user_id"] = message.user_id
            if call.name == "search_knowledge" and "query" not in args:
                args["query"] = message.content
            if call.name == "execute_in_sandbox":
                run_id = uuid.uuid4().hex
                task = args.get("task") or args.get("description") or args.get("query")
                if not task:
                    task = self._build_task_from_args(args)
                code = args.get("code")
                if not code:
                    code = self._generate_sandbox_code(
                        task=task,
                        inputs=args.get("inputs"),
                        results=results,
                    )
                code = self._build_sandbox_code(
                    code=code,
                    task=task,
                    inputs=args.get("inputs"),
                    results=results,
                )
                logger.info(
                    "============== [LLM GENERATED CODE] ==============\n%s\n==================================================",
                    code,
                )
                self._validate_code(code)
                required_packages = args.get("required_packages", []) or []
                inferred_packages = self._infer_packages_from_code(code)
                if inferred_packages:
                    required_packages = self._ensure_packages(required_packages, inferred_packages)
                if self._needs_plot_packages(message.content):
                    required_packages = self._ensure_packages(required_packages, ["matplotlib"])

                ## 코드 실행
                try:
                    sandbox_result = self.sandbox_client.run_code(
                        code=code,
                        required_packages=required_packages,
                        user_id=message.user_id,
                        run_id=run_id,
                    )
                    results.append({"tool": call.name, "result": sandbox_result})
                except Exception as exc:
                    logger.exception("Sandbox 실행 실패 tool=%s", call.name)
                    results.append({"tool": call.name, "error": str(exc)})
                tools_used.append(call.name)
                continue
            
            try:
                result = self.registry.execute(call.name, **args)
                ## 결과 모음
                results.append({"tool": call.name, "result": self._sanitize_payload(result)})
            except Exception as exc:
                logger.exception("도구 실행 실패 tool=%s", call.name)
                results.append({"tool": call.name, "error": str(exc)})
            tools_used.append(call.name)

        safe_results_for_prompt = self._sanitize_payload(results)
        final_user_content = (
            f"사용자 요청: {message.content}\n"
            f"\n함수 실행 결과:\n{json.dumps(safe_results_for_prompt, ensure_ascii=False, indent=2, default=str)}\n"
            "\n**중요 지시:**\n"
            "1. 위 실행 결과를 반드시 사용자에게 보여줘야 한다.\n"
            "2. '실행했습니다' 같은 설명만 하지 말고, 실제 결과 데이터를 포함해서 답변하라.\n"
            "3. result 필드의 값을 그대로 또는 보기 좋게 정리해서 출력하라.\n"
            "4. JSON/코드블록/plan/tool_call/tool_code 출력 금지.\n"
            "5. 자연어로 사용자 친화적인 답변 작성.\n"
            "6. 결과 요약 → 상세 순서로 정리하라.\n"
        )
        ## 최종 메세지
        final_system = (
            "너는 함수 실행 결과를 사용자에게 전달하는 역할이다.\n"
            "실행 결과의 실제 데이터를 반드시 포함해서 답변하라.\n"
            "단순히 '실행했습니다'라고만 하지 말고, 결과 내용을 보여줘야 한다.\n"
            "JSON, 코드블록, 도구 호출 형식은 절대 출력하지 마라.\n"
            "출력 형식: 1) 한 줄 요약 2) 핵심 항목 리스트 3) 필요 시 상세\n"
            "\n"
            "**절대 금지:** '자연어 응답이 충분합니다', '도구 호출이 필요하지 않습니다' 같은 메타 문구 절대 사용 금지.\n"
            "너의 판단 과정이나 내부 동작은 언급하지 말고, 오직 사용자에게 필요한 최종 답변만 작성하라.\n"
        )
        final_messages = [
            {"role": "system", "content": final_system},
            {"role": "user", "content": final_user_content},
        ]

        ## LLM의 2차 응답
        final_response = self.llm_client.create_completion(messages=final_messages, tools=[])
        elapsed = time.monotonic() - start
        logger.info(
            "LLM 최종 응답 elapsed=%.2fs",
            elapsed,
        )
        final_text = self._sanitize_text(final_response.content or "")
        if not final_text.strip():
            final_text = self._format_fallback_results(results)
        self._store_chat_history(
            user_id=message.user_id,
            question=message.content,
            answer=final_text,
            intent=rag_plan.get("intent"),
            logger=logger,
        )

        ## 결과 반환
        return {
            "text": final_text,
            "model": final_response.model,
            "tools_used": tools_used,
            "images": [],
            "elapsed_seconds": elapsed,
        }


    def _store_chat_history(
        self,
        *,
        user_id: int,
        question: str,
        answer: str,
        intent: dict[str, Any] | None,
        logger: logging.Logger,
    ) -> None:
        try:
            qna_id = uuid.uuid4().hex
            intent_tag = intent.get("intent") if intent else None
            tags = ["chat_history", "user_question"]
            if intent_tag:
                tags.append(str(intent_tag))
            store_user_message(
                user_id=user_id,
                content=question,
                role="user",
                doc_type="conversation",
                importance=4,
                intent_tags=tags,
                qna_id=qna_id,
            )
            if answer:
                store_user_message(
                    user_id=user_id,
                    content=answer,
                    role="assistant",
                    doc_type="assistant_reply",
                    importance=2,
                    intent_tags=["chat_history", "assistant_reply"],
                    qna_id=qna_id,
                )
        except Exception:
            logger.exception("MongoDB 대화 저장 실패")

    def _generate_sandbox_code(
        self,
        task: str | None,
        inputs: dict[str, Any] | None,
        results: list[dict[str, Any]],
    ) -> str:
        if not task:
            return "import json\nprint(json.dumps(inputs, ensure_ascii=False))"

        # LLM이 모든 작업을 판단하고 코드 생성
        system_prompt = (
            "너는 Python 코드 생성기다.\n"
            "사용자 요청을 보고 적절한 Python 코드를 생성한다.\n"
            "변수 inputs(dict)는 이미 존재한다.\n"
            "\n"
            "**핵심 원칙:**\n"
            "1. 명령어처럼 보이는 요청 → subprocess.run()으로 실행\n"
            "2. 계산/분석 요청 → 직접 Python 코드로 구현\n"
            "3. 반드시 결과를 print()로 출력\n"
            "4. subprocess 사용 시 stdout/stderr을 반드시 print\n"
            "5. 설명/주석 없이 실행 가능한 코드만 출력\n"
            "\n"
            "**중요: subprocess 템플릿 (명령어 실행용)**\n"
            "```python\n"
            "import subprocess\n"
            "result = subprocess.run('실제명령어', shell=True, capture_output=True, text=True)\n"
            "output = result.stdout.strip() if result.stdout else result.stderr.strip()\n"
            "print(output if output else '[출력 없음]')\n"
            "```\n"
            "\n"
            "**예시:**\n"
            "\n"
            "요청: grep CapEff /proc/self/status\n"
            "코드:\n"
            "```python\n"
            "import subprocess\n"
            "result = subprocess.run('grep CapEff /proc/self/status', shell=True, capture_output=True, text=True)\n"
            "output = result.stdout.strip() if result.stdout else result.stderr.strip()\n"
            "print(output if output else '[출력 없음]')\n"
            "```\n"
            "\n"
            "요청: ls -la /tmp\n"
            "코드:\n"
            "```python\n"
            "import subprocess\n"
            "result = subprocess.run('ls -la /tmp', shell=True, capture_output=True, text=True)\n"
            "output = result.stdout.strip() if result.stdout else result.stderr.strip()\n"
            "print(output if output else '[출력 없음]')\n"
            "```\n"
            "\n"
            "요청: 1부터 100까지 합 계산\n"
            "코드:\n"
            "```python\n"
            "total = sum(range(1, 101))\n"
            "print(f'1부터 100까지의 합: {total}')\n"
            "```\n"
            "\n"
            "요청: cat /etc/os-release\n"
            "코드:\n"
            "```python\n"
            "import subprocess\n"
            "result = subprocess.run('cat /etc/os-release', shell=True, capture_output=True, text=True)\n"
            "output = result.stdout.strip() if result.stdout else result.stderr.strip()\n"
            "print(output if output else '[출력 없음]')\n"
            "```\n"
        )
        
        payload = inputs if inputs is not None else {"results": results, "task": task}
        user_prompt = (
            f"요청: {task}\n"
            f"inputs: {json.dumps(payload, ensure_ascii=False)}\n"
            "\n"
            "위 요청을 수행하는 Python 코드를 생성하라.\n"
            "명령어처럼 보이면 subprocess.run()을 사용하고,\n"
            "계산/분석이면 직접 구현하라.\n"
            "반드시 결과를 print()로 출력해야 한다.\n"
            "코드만 출력하고 설명은 하지 마라."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.llm_client.create_completion(messages=messages, tools=[])
        raw_code = response.content or ""
        return self._strip_code_fences(raw_code).strip() or "import json\nprint(json.dumps(inputs, ensure_ascii=False))"

    def _build_task_from_args(self, args: dict[str, Any]) -> str | None:
        title = args.get("title")
        vis_type = args.get("visualization_type") or args.get("type")
        x_label = args.get("x_axis_label") or args.get("x_axis")
        y_label = args.get("y_axis_label") or args.get("y_axis")
        data_source = args.get("data_source") or args.get("data")
        parts = []
        if vis_type:
            parts.append(f"{vis_type} 그래프")
        if title:
            parts.append(f"제목: {title}")
        if x_label:
            parts.append(f"x축: {x_label}")
        if y_label:
            parts.append(f"y축: {y_label}")
        if data_source:
            parts.append(f"데이터: {data_source}")
        return " / ".join(parts) if parts else None

    def _parse_args(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return self._normalize_params(arguments)
        if isinstance(arguments, str):
            parsed: Any = json.loads(arguments)
            if isinstance(parsed, str):
                trimmed = parsed.strip()
                if (trimmed.startswith("{") and trimmed.endswith("}")) or (
                    trimmed.startswith("[") and trimmed.endswith("]")
                ):
                    try:
                        parsed = json.loads(trimmed)
                    except Exception:
                        return {}
            return self._normalize_params(parsed) if isinstance(parsed, dict) else {}
        raise ValueError("Tool arguments 형식이 올바르지 않습니다.")

    def _extract_tool_calls(self, content: str) -> list[Any]:
        """
        LLM이 tool_calls 대신 텍스트로 도구 호출을 작성하는 경우를 파싱한다.
        지원 형식:
        - ```tool_code\nget_user_profile(user_id=13)\n```
        - ```json\n{"actions":[{"function":"get_user_profile","parameters":{"user_id":13}}]}\n```
        """
        tool_calls: list[Any] = []

        for match in _TOOL_CODE_PATTERN.findall(content):
            stripped = match.strip()
            if stripped.startswith("["):
                tool_calls.extend(self._parse_plan(stripped))
                continue
            # tool_code 블록은 "도구 호출 + 파이썬 코드"가 섞여 나올 수 있다.
            # 예: execute_in_sandbox(task=\"\"\"...\nprint(...)\n\"\"\") 형태
            # → 블록 전체를 우선 단일 도구 호출로 파싱하고, 실패 시에만 라인 단위 파싱을 시도한다.
            block_calls = self._parse_tool_code_block(stripped)
            if block_calls:
                tool_calls.extend(block_calls)
                continue

            lines = [line.strip() for line in match.splitlines() if line.strip()]
            for line in lines:
                name, args = self._parse_tool_code_line(line)
                if name and self._is_known_tool_name(name):
                    tool_calls.append(SimpleNamespace(name=name, arguments=args))

        content_lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in content_lines:
            if re.match(r"^[a-zA-Z_][\w]*\s*\(.*\)\s*$", line):
                name, args = self._parse_tool_code_line(line)
                if name and self._is_known_tool_name(name):
                    tool_calls.append(SimpleNamespace(name=name, arguments=args))

        for match in _TOOL_CALL_FENCE_PATTERN.findall(content):
            try:
                payload = json.loads(match)
            except Exception:
                continue
            if isinstance(payload, dict):
                name = payload.get("tool") or payload.get("name")
                args = payload.get("arguments") or payload.get("params") or {}
                if not args:
                    args = {}
                    for key in ("task", "code", "inputs", "required_packages"):
                        if key in payload:
                            args[key] = payload[key]
                if name:
                    tool_calls.append(SimpleNamespace(name=name, arguments=args))

        for match in _TOOL_CALLS_FENCE_PATTERN.findall(content):
            # 예: ```tool_calls\n[{"type":"function","function":"execute_in_sandbox","arguments":{"task":"..."}}]\n```
            tool_calls.extend(self._parse_plan(match))

        for match in _ACTIONS_JSON_PATTERN.findall(content):
            tool_calls.extend(self._parse_plan(match))

        for match in _JSON_FENCE_PATTERN.findall(content):
            tool_calls.extend(self._parse_plan(match))

        for payload in self._extract_json_payloads(content):
            tool_calls.extend(self._parse_function_payload(payload))

        content_stripped = content.strip()
        if content_stripped.startswith("{") and content_stripped.endswith("}"):
            tool_calls.extend(self._parse_plan(content_stripped))
        if not tool_calls and "tool_call" in content_stripped:
            stripped = self._strip_code_fences(content_stripped).strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                tool_calls.extend(self._parse_plan(stripped))

        return tool_calls

    def _is_known_tool_name(self, name: str) -> bool:
        return bool(_KNOWN_TOOL_NAME_PATTERN.match((name or "").strip()))

    def _parse_tool_code_block(self, block: str) -> list[Any]:
        """
        tool_code 펜스 내부 텍스트 전체를 '단일 도구 호출'로 파싱한다.
        특히 execute_in_sandbox(task=\"\"\"...\"\"\") 같은 멀티라인 인자를 안전하게 처리한다.
        """
        text = (block or "").strip()
        if not text:
            return []

        # 전체가 하나의 함수 호출 형태인지 확인 (멀티라인 허용)
        m = re.match(r"^\s*([a-zA-Z_]\w*)\s*\((.*)\)\s*$", text, flags=re.DOTALL)
        if not m:
            return []
        name = m.group(1).strip()
        args_blob = m.group(2) or ""

        if not self._is_known_tool_name(name):
            return []

        # 멀티라인 task=...만 특별 처리
        if name == "execute_in_sandbox":
            tm = re.search(
                r"task\s*=\s*(\"\"\"|'''|\"|')(?P<body>.*?)(?:\1)",
                args_blob,
                flags=re.DOTALL,
            )
            task = (tm.group("body") if tm else "").strip()
            if not task:
                return []
            return [SimpleNamespace(name=name, arguments={"task": task})]

        # 그 외는 기존 단일 라인 파서로 처리(실패하면 무시)
        if "\n" in args_blob:
            return []
        parsed_name, parsed_args = self._parse_tool_code_line(f"{name}({args_blob})")
        if parsed_name and self._is_known_tool_name(parsed_name):
            return [SimpleNamespace(name=parsed_name, arguments=parsed_args)]
        return []


    def _parse_plan(self, raw: str) -> list[Any]:
        try:
            data = json.loads(raw)
        except Exception:
            return []

        tool_calls: list[Any] = []
        if isinstance(data, str):
            return tool_calls
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                tool_call_payload = item.get("tool_call")
                if isinstance(tool_call_payload, dict):
                    function_payload = tool_call_payload.get("function")
                    if isinstance(function_payload, dict):
                        name = function_payload.get("name")
                        params = function_payload.get("arguments") or {}
                    else:
                        name = tool_call_payload.get("tool") or tool_call_payload.get("name")
                        params = (
                            tool_call_payload.get("parameters")
                            or tool_call_payload.get("params")
                            or tool_call_payload.get("arguments")
                            or {}
                        )
                else:
                    function_payload = item.get("function")
                    if isinstance(function_payload, dict):
                        name = function_payload.get("name")
                        params = function_payload.get("arguments") or {}
                    else:
                        name = item.get("tool") or item.get("function") or item.get("name")
                        params = item.get("parameters") or item.get("params") or item.get("arguments") or {}
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {}
                if name:
                    tool_calls.append(
                        SimpleNamespace(name=name, arguments=self._normalize_params(params))
                    )
            return tool_calls
        if not isinstance(data, dict):
            return tool_calls

        actions = data.get("tool_calls") or data.get("actions") or data.get("plan") or []
        if isinstance(actions, dict):
            actions = actions.get("steps", [])
        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("action") == "execute_in_sandbox":
                tool_calls.append(
                    SimpleNamespace(
                        name="execute_in_sandbox",
                        arguments=self._normalize_params({"task": action.get("task")}),
                    )
                )
                continue
            function_payload = action.get("function")
            if isinstance(function_payload, dict):
                name = function_payload.get("name")
                params = function_payload.get("arguments") or {}
            else:
                name = (
                    action.get("tool")
                    or action.get("function")
                    or action.get("function_name")
                    or action.get("name")
                )
                params = action.get("parameters") or action.get("params") or action.get("arguments") or {}
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    params = {}
            if name:
                tool_calls.append(
                    SimpleNamespace(name=name, arguments=self._normalize_params(params))
                )
        return tool_calls

    def _parse_function_payload(self, payload: Any) -> list[Any]:
        tool_calls: list[Any] = []

        def _add(name: str | None, arguments: Any) -> None:
            if not name:
                return
            args = arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(
                SimpleNamespace(name=name, arguments=self._normalize_params(args))
            )

        def _visit(node: Any) -> None:
            if isinstance(node, list):
                for item in node:
                    _visit(item)
                return
            if not isinstance(node, dict):
                return

            if "tool_calls" in node and isinstance(node["tool_calls"], list):
                for call in node["tool_calls"]:
                    if isinstance(call, dict):
                        function = call.get("function") or {}
                        if isinstance(function, dict):
                            _add(function.get("name"), function.get("arguments"))
                        else:
                            _add(call.get("name"), call.get("arguments"))
                return

            if "function_call" in node and isinstance(node["function_call"], dict):
                _add(node["function_call"].get("name"), node["function_call"].get("arguments"))
                return

            if "function" in node and isinstance(node["function"], dict):
                _add(node["function"].get("name"), node["function"].get("arguments"))
                return

            if "name" in node and "arguments" in node:
                _add(node.get("name"), node.get("arguments"))
                return

            for value in node.values():
                _visit(value)

        _visit(payload)
        return tool_calls

    def _extract_json_payloads(self, content: str) -> list[Any]:
        if not content:
            return []
        decoder = json.JSONDecoder()
        payloads: list[Any] = []
        idx = 0
        length = len(content)
        while idx < length:
            if content[idx] not in {"{", "["}:
                idx += 1
                continue
            try:
                obj, end = decoder.raw_decode(content[idx:])
            except Exception:
                idx += 1
                continue
            payloads.append(obj)
            idx += max(end, 1)
        return payloads

    def _parse_tool_code_line(self, line: str) -> tuple[str | None, dict[str, Any]]:
        if "(" not in line or not line.endswith(")"):
            return None, {}
        
        try:
            name, raw_args = line.split("(", 1)
            name = name.strip()
            raw_args = raw_args[:-1].strip()
            
            if not raw_args:
                return name, {}
            
            args: dict[str, Any] = {}
            
            # 더 정교한 파싱: 따옴표 안의 comma를 무시
            current_key = ""
            current_value = ""
            in_quotes = False
            quote_char = None
            parsing_value = False
            
            i = 0
            while i < len(raw_args):
                char = raw_args[i]
                
                # 따옴표 처리
                if char in ('"', "'") and (i == 0 or raw_args[i-1] != "\\"):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                
                # = 발견: key -> value 전환
                elif char == "=" and not in_quotes and not parsing_value:
                    current_key = current_key.strip()
                    parsing_value = True
                
                # comma 발견: 다음 파라미터로
                elif char == "," and not in_quotes:
                    if current_key and parsing_value:
                        value = current_value.strip().strip('"').strip("'")
                        # 타입 변환
                        if value.isdigit():
                            args[current_key] = int(value)
                        elif value.lower() in ("true", "false"):
                            args[current_key] = value.lower() == "true"
                        else:
                            try:
                                args[current_key] = float(value)
                            except ValueError:
                                args[current_key] = value
                    current_key = ""
                    current_value = ""
                    parsing_value = False
                
                # 문자 누적
                else:
                    if parsing_value:
                        current_value += char
                    else:
                        current_key += char
                
                i += 1
            
            # 마지막 파라미터 처리
            if current_key and parsing_value:
                value = current_value.strip().strip('"').strip("'")
                if value.isdigit():
                    args[current_key] = int(value)
                elif value.lower() in ("true", "false"):
                    args[current_key] = value.lower() == "true"
                else:
                    try:
                        args[current_key] = float(value)
                    except ValueError:
                        args[current_key] = value
            
            return name, args
            
        except Exception:
            # 파싱 실패 시 기본값 반환
            return None, {}

    def _normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        def _normalize_key(key: str) -> str:
            compact = key.replace("_", "").lower()
            if compact == "userid":
                return "user_id"
            if compact in {"desc", "description", "query"}:
                return "task"
            return key

        def _normalize_value(value: Any) -> Any:
            if isinstance(value, dict):
                return { _normalize_key(k): _normalize_value(v) for k, v in value.items() }
            if isinstance(value, list):
                return [_normalize_value(item) for item in value]
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed.isdigit():
                    return int(trimmed)
                try:
                    return float(trimmed)
                except ValueError:
                    return value
            return value

        return { _normalize_key(k): _normalize_value(v) for k, v in params.items() }

    def _filter_tools_by_allowlist(self, tools: list[dict], allowlist: list[str]) -> list[dict]:
        if not allowlist:
            return tools
        allowset = set(allowlist)
        filtered: list[dict] = []
        for item in tools:
            name = item.get("function", {}).get("name")
            if name == "execute_in_sandbox" or name in allowset:
                filtered.append(item)
        return filtered

    def _strip_code_fences(self, text: str) -> str:
        if not text:
            return text
        cleaned = re.sub(r"```(?:python)?\s*", "", text, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "")
        return cleaned

    def _build_sandbox_code(
        self,
        code: str | None,
        task: str | None,
        inputs: dict[str, Any] | None,
        results: list[dict[str, Any]],
    ) -> str:
        payload = inputs if inputs is not None else {"results": results, "task": task}
        payload = self._sanitize_payload(payload)
        encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        prelude = (
            "import base64\n"
            "import json\n"
            "import os\n"
            "import subprocess\n"  # subprocess 추가
            "import sys\n"  # sys 추가
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            f"inputs = json.loads(base64.b64decode('{base64.b64encode(encoded).decode('ascii')}').decode('utf-8'))\n"
        )
        if code:
            return f"{prelude}\n{code}"
        return f"{prelude}\nprint(json.dumps(inputs, ensure_ascii=False))"

    def _validate_code(self, code: str) -> None:
        # Sandbox Evasion 취약점: 코드 검증 완화
        vulnerable_sandbox = os.getenv("VULNERABLE_SANDBOX_EVASION", "false").strip().lower() in {"true", "1", "yes"}
        
        if vulnerable_sandbox:
            return  # 모든 코드 허용
        
        # 기본 모드에서는 검증 비활성화 (교육 목적)
        pass
        # pattern = globals().get("_BLOCKED_CODE_PATTERN")
        # if pattern and pattern.search(code):
        #     raise ValueError("Sandbox 코드에 금지된 키워드가 포함되어 있습니다.")

    def _sanitize_payload(self, payload: Any) -> Any:
        # Sensitive Information Disclosure 취약점: 민감정보 필터링 완화
        vulnerable_disclosure = os.getenv("VULNERABLE_SENSITIVE_DISCLOSURE", "false").strip().lower() in {"true", "1", "yes"}

        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, bytes):
            try:
                return payload.decode("utf-8", errors="replace")
            except Exception:
                return str(payload)
        if isinstance(payload, dict):
            if vulnerable_disclosure:
                return {k: self._sanitize_payload(v) for k, v in payload.items()}  # 민감 키 필터링 안 함
            return {k: self._sanitize_payload(v) for k, v in payload.items() if k not in _SENSITIVE_KEYS}
        if isinstance(payload, list):
            return [self._sanitize_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return [self._sanitize_payload(item) for item in payload]
        if isinstance(payload, set):
            return [self._sanitize_payload(item) for item in payload]

        # bson.ObjectId, datetime 등 JSON 비직렬화 객체 처리
        class_name = payload.__class__.__name__
        if class_name in {"ObjectId", "datetime", "date"}:
            return str(payload)

        # Pydantic/model-like 객체 처리
        if hasattr(payload, "model_dump"):
            try:
                return self._sanitize_payload(payload.model_dump())
            except Exception:
                return str(payload)
        if hasattr(payload, "dict"):
            try:
                return self._sanitize_payload(payload.dict())
            except Exception:
                return str(payload)
        if hasattr(payload, "__dict__"):
            try:
                return self._sanitize_payload(vars(payload))
            except Exception:
                return str(payload)
        return str(payload)

    def _sanitize_text(self, text: str) -> str:
        if not text:
            return text
        
        # Sensitive Information Disclosure 취약점: 민감정보 마스킹 완화
        vulnerable_disclosure = os.getenv("VULNERABLE_SENSITIVE_DISCLOSURE", "false").strip().lower() in {"true", "1", "yes"}
        
        text = re.sub(
            r"```tool_call\s*[\s\S]*?```",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"```tool_code\s*[\s\S]*?```",
            "",
            text,
            flags=re.IGNORECASE,
        )
        
        if not vulnerable_disclosure:
            for key in _SENSITIVE_KEYS:
                text = re.sub(fr"{key}\s*:\s*\S+", f"{key}: ***", text, flags=re.IGNORECASE)
        
        # 연속된 공백과 줄바꿈 정리
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r"^\s+", "", text)
        
        return text.strip()

    def _is_direct_prompt_request(self, text: str | None) -> bool:
        """직접적/명시적 시스템 프롬프트 요청 탐지 (항상 차단)"""
        if not text:
            return False
        return bool(_SYSTEM_PROMPT_DIRECT_PATTERN.search(text))

    def _is_indirect_prompt_request(self, text: str | None) -> bool:
        """간접적/우회 시스템 프롬프트 탈취 시도 탐지"""
        if not text:
            return False
        return bool(_SYSTEM_PROMPT_INDIRECT_PATTERN.search(text))


    def _format_fallback_results(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "요청 결과가 비어 있어 표시할 내용이 없습니다."
        lines = ["요청 결과를 정리했습니다."]
        for item in results:
            tool_name = item.get("tool") or "도구"
            summary = self._summarize_result(item.get("result"))
            lines.append(f"{tool_name}: {summary}")
        return "\n".join(lines)

    def _summarize_result(self, result: Any) -> str:
        if result is None:
            return "결과 없음"
        if isinstance(result, str):
            trimmed = result.strip()
            return trimmed or "결과 없음"
        if isinstance(result, list):
            if not result:
                return "목록이 비어 있습니다."
            if all(isinstance(item, dict) for item in result):
                items_text = []
                for idx, item in enumerate(result[:3], start=1):
                    parts = []
                    for key, value in list(item.items())[:6]:
                        parts.append(f"{key}={self._simple_value(value)}")
                    items_text.append(f"{idx}) " + ", ".join(parts))
                more = "" if len(result) <= 3 else f" 외 {len(result) - 3}건"
                return "; ".join(items_text) + more
            if all(not isinstance(item, (dict, list)) for item in result):
                preview = ", ".join(str(item) for item in result[:8])
                more = "" if len(result) <= 8 else f" 외 {len(result) - 8}건"
                return f"{preview}{more}"
            return f"목록 {len(result)}건"
        if isinstance(result, dict):
            if not result:
                return "결과 없음"
            parts = []
            for key, value in list(result.items())[:8]:
                parts.append(f"{key}={self._simple_value(value)}")
            more = "" if len(result) <= 8 else " 외 추가 항목"
            return ", ".join(parts) + more
        return str(result)

    def _simple_value(self, value: Any) -> str:
        if value is None:
            return "없음"
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, str):
            return value.strip() or "없음"
        if isinstance(value, list):
            return f"목록 {len(value)}건"
        if isinstance(value, dict):
            return f"객체 {len(value)}항목"
        return str(value)

    def _needs_plot_packages(self, text: str) -> bool:
        return bool(_PLOT_KEYWORDS_PATTERN.search(text or ""))

    def _ensure_packages(self, packages: list[str], required: list[str]) -> list[str]:
        normalized = {pkg.lower() for pkg in packages}
        merged = list(packages)
        for pkg in required:
            if pkg.lower() not in normalized:
                merged.append(pkg)
                normalized.add(pkg.lower())
        return merged

    def _infer_packages_from_code(self, code: str) -> list[str]:
        if not code:
            return []
        candidates: list[str] = []
        for match in _IMPORT_PATTERN.findall(code):
            module = match.split(".", 1)[0].strip().lower()
            if module in _AUTO_PACKAGE_ALLOWLIST:
                candidates.append(module)
        if not candidates:
            return []
        # preserve order, de-duplicate
        seen: set[str] = set()
        ordered: list[str] = []
        for item in candidates:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered
