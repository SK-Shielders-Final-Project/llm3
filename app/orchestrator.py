from __future__ import annotations

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


_BLOCKED_CODE_PATTERN = re.compile(
    r"(import\s+sys|subprocess|socket|requests|shutil|rm\s+-rf|"
    r"os\.system|__import__|open\(|eval\(|exec\()",
    re.IGNORECASE,
)

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

        rag_plan = self.rag_pipeline.plan_tool_selection(
            question=message.content,
            user_id=message.user_id,
            admin_level=getattr(message, "admin_level", None),
        )
        decision = rag_plan.get("decision") or {}
        if decision.get("data_source") == "vector_only":
            rag_result = self.rag_pipeline.process_question(
                question=message.content,
                user_id=message.user_id,
                admin_level=getattr(message, "admin_level", None),
            )
            return {
                "text": rag_result.get("answer", ""),
                "model": "rag_pipeline",
                "tools_used": [],
                "images": [],
            }

        ## 시스템 프롬프트 주입
        system_prompt = build_system_context(message)
        ## 해당 도구 사용하는 스키마
        tools = self._filter_tools_by_allowlist(
            build_tool_schema(), rag_plan.get("tool_allowlist", [])
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{message.content}\n\n{rag_plan.get('context','')}"},
        ]

        ## llm 실행 1차 응답
        response = self.llm_client.create_completion(messages=messages, tools=tools)
        ## LLM이 Tool을 요청하거나 Plan JSON형식으로 전달

        logger.info(
            "LLM 1차 응답 elapsed=%.2fs tool_calls=%s",
            time.monotonic() - start,
            len(response.tool_calls),
        )

        ## 다른 도구들 실행 여부 확인
        tool_calls = response.tool_calls or self._extract_tool_calls(response.content or "")

        if not tool_calls:
            fallback_text = self._sanitize_text(response.content or "")
            logger.warning(
                "LLM tool_calls 누락: fallback_response_used=%s content=%s",
                bool(fallback_text),
                fallback_text,
            )
            if fallback_text:
                self._store_chat_history(
                    user_id=message.user_id,
                    question=message.content,
                    answer=fallback_text,
                    intent=rag_plan.get("intent"),
                    logger=logger,
                )
                return {
                    "text": fallback_text,
                    "model": response.model,
                    "tools_used": [],
                    "images": [],
                }
            raise ValueError("LLM이 tool_calls 또는 plan JSON을 반환하지 않았습니다.")


        ## 결과, 사용된 도구를 배열로 담음
        results: list[dict[str, Any]] = []
        tools_used: list[str] = []

        ## 도구 실행 루프
        for call in tool_calls:
            args = self._parse_args(call.arguments)
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
                    code = self._generate_sandbox_code(task=task, inputs=args.get("inputs"), results=results)
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
                sandbox_result = self.sandbox_client.run_code(
                    code=code,
                    required_packages=required_packages,
                    user_id=message.user_id,
                    run_id=run_id,
                )

                results.append({"tool": call.name, "result": sandbox_result})
                tools_used.append(call.name)
                continue
            
            result = self.registry.execute(call.name, **args)
            ## 결과 모음
            results.append({"tool": call.name, "result": self._sanitize_payload(result)})
            tools_used.append(call.name)

        final_user_content = (
            f"사용자 요청: {message.content}\n"
            f"라우팅 컨텍스트:\n{rag_plan.get('context','')}\n"
            "이제 도구 호출은 금지된다. plan/json/tool_code를 출력하지 말고 "
            "최종 사용자 답변만 자연어로 작성하라.\n"
            f"함수 실행 결과: {json.dumps(results, ensure_ascii=False)}"
        )
        ## 최종 메세지
        final_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user_content},
        ]

        ## LLM의 2차 응답
        final_response = self.llm_client.create_completion(messages=final_messages, tools=tools)
        logger.info(
            "LLM 최종 응답 elapsed=%.2fs",
            time.monotonic() - start,
        )
        final_text = self._sanitize_text(final_response.content or "")
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
            )
            if answer:
                store_user_message(
                    user_id=user_id,
                    content=answer,
                    role="assistant",
                    doc_type="assistant_reply",
                    importance=2,
                    intent_tags=["chat_history", "assistant_reply"],
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

        system_prompt = (
            "너는 Python 코드 생성기다. "
            "이미 변수 inputs(dict)가 존재한다고 가정하고 이를 활용한다. "
            "설명/주석/코드블록 없이 Python 코드만 출력하라."
        )
        payload = inputs if inputs is not None else {"results": results, "task": task}
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"작업: {task}\n"
                    f"inputs: {json.dumps(payload, ensure_ascii=False)}"
                ),
            },
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
            lines = [line.strip() for line in match.splitlines() if line.strip()]
            for line in lines:
                name, args = self._parse_tool_code_line(line)
                if name:
                    tool_calls.append(SimpleNamespace(name=name, arguments=args))

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
                    name = (
                        tool_call_payload.get("tool")
                        or tool_call_payload.get("function")
                        or tool_call_payload.get("name")
                    )
                    params = (
                        tool_call_payload.get("parameters")
                        or tool_call_payload.get("params")
                        or tool_call_payload.get("arguments")
                        or {}
                    )
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
        name, raw_args = line.split("(", 1)
        name = name.strip()
        raw_args = raw_args[:-1].strip()
        if not raw_args:
            return name, {}
        args: dict[str, Any] = {}
        for pair in raw_args.split(","):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value.isdigit():
                args[key] = int(value)
            else:
                try:
                    args[key] = float(value)
                except ValueError:
                    args[key] = value
        return name, args

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
        encoded = json.dumps(payload, ensure_ascii=False)
        prelude = (
            "import json\n"
            "import os\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            f"inputs = json.loads('''{encoded}''')\n"
        )
        if code:
            return f"{prelude}\n{code}"
        return f"{prelude}\nprint(json.dumps(inputs, ensure_ascii=False))"

    def _validate_code(self, code: str) -> None:
        if _BLOCKED_CODE_PATTERN.search(code):
            raise ValueError("Sandbox 코드에 금지된 키워드가 포함되어 있습니다.")

    def _sanitize_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {k: self._sanitize_payload(v) for k, v in payload.items() if k not in _SENSITIVE_KEYS}
        if isinstance(payload, list):
            return [self._sanitize_payload(item) for item in payload]
        return payload

    def _sanitize_text(self, text: str) -> str:
        for key in _SENSITIVE_KEYS:
            text = re.sub(fr"{key}\s*:\s*\S+", f"{key}: ***", text, flags=re.IGNORECASE)
        return text

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
