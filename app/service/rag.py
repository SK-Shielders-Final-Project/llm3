from __future__ import annotations

import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from app.clients.guardrail_client import GuardrailClient
from app.clients.llm_client import LlmClient
from app.service.mongo.search import search_knowledge
from app.service.mongo.store import store_user_message
from app.service.mongo.manual_search import (
    search_manual_knowledge,
    format_manual_results_as_context,
)
from app.service.registry import (
    get_available_bikes,
    get_inquiries,
    get_payments,
    get_rentals,
    get_total_payments,
    get_total_usage,
    get_user_profile,
)

logger = logging.getLogger(__name__)

IntentType = Literal["personal_data", "general_knowledge", "realtime_location", "admin_function"]
DataSourceType = Literal["vector_only", "mysql_only", "hybrid"]


class KnowledgeMetadata(TypedDict, total=False):
    doc_type: str
    category: str
    requires_mysql: bool
    mysql_tables: list[str]
    access_level: str
    requires_auth: bool
    importance: int
    freshness_score: float
    intent_tags: list[str]
    created_at: str
    updated_at: str
    user_id: int
    feedback_score: float
    topic_tags: list[str]
    entities: list[str]
    session_id: str
    turn_index: int


class KnowledgeDocument(TypedDict, total=False):
    content: str
    metadata: KnowledgeMetadata
    score: float
    final_score: float


@dataclass(frozen=True)
class RagWeights:
    llm_confidence: float = 0.55
    vector_confidence: float = 0.25
    mysql_signal: float = 0.1
    history_signal: float = 0.1

    intent_boost: float = 0.2
    importance_boost: float = 0.15
    freshness_boost: float = 0.1
    feedback_boost: float = 0.2
    recency_boost: float = 0.15

    vector_force_threshold: float = 0.82
    mysql_force_threshold: float = 0.7
    hybrid_threshold: float = 0.55


class RagPipeline:
    def __init__(
        self,
        llm_client: LlmClient,
        weights: RagWeights | None = None,
        guardrail_client: GuardrailClient | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.weights = weights or RagWeights()
        self.guardrail_client = guardrail_client

    def process_question(
        self,
        question: str,
        user_id: int,
        admin_level: int | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        guarded_question = self._apply_guardrail_input(question)
        admin_level = self._resolve_admin_level(user_id, admin_level)
        intent = self._classify_intent(guarded_question)
        raw_docs = search_knowledge(
            query=guarded_question,
            user_id=user_id,
            admin_level=admin_level,
            top_k=top_k,
        )
        docs = self._apply_vector_boosts(raw_docs, intent)
        decision = self._decide_data_source(intent, docs)
        mysql_data = self._fetch_mysql_data(
            intent=intent, user_id=user_id, admin_level=admin_level, decision=decision
        )
        answer = self._generate_answer(guarded_question, intent, docs, mysql_data)
        self._store_conversation(
            user_id=user_id, question=guarded_question, answer=answer, intent=intent
        )
        return {
            "answer": answer,
            "intent": intent,
            "decision": decision,
            "vector_docs": docs,
            "mysql_data": mysql_data,
        }

    def answer_from_plan(
        self,
        *,
        question: str,
        user_id: int,
        plan: dict[str, Any],
        admin_level: int | None = None,
    ) -> dict[str, Any]:
        """
        plan_tool_selection에서 생성된 결과를 재사용해 답변을 만든다.
        vector_only 경로에서 중복 검색/분류를 줄이기 위해 사용한다.
        """
        guarded_question = self._apply_guardrail_input(question)
        intent = plan.get("intent") or {}
        docs = plan.get("vector_docs") or []
        decision = plan.get("decision") or {}
        resolved_admin = self._resolve_admin_level(user_id, admin_level)

        mysql_data: dict[str, Any] | None = None
        if decision.get("data_source") in {"mysql_only", "hybrid"}:
            mysql_data = self._fetch_mysql_data(
                intent=intent,
                user_id=user_id,
                admin_level=resolved_admin,
                decision=decision,
            )

        answer = self._generate_answer(guarded_question, intent, docs, mysql_data)
        self._store_conversation(
            user_id=user_id, question=guarded_question, answer=answer, intent=intent
        )
        return {
            "answer": answer,
            "intent": intent,
            "decision": decision,
            "vector_docs": docs,
            "mysql_data": mysql_data,
        }

    def plan_tool_selection(
        self,
        question: str,
        user_id: int,
        admin_level: int | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        guarded_question = self._apply_guardrail_input(question)
        admin_level = self._resolve_admin_level(user_id, admin_level)
        intent = self._classify_intent(guarded_question)
        raw_docs = search_knowledge(
            query=guarded_question,
            user_id=user_id,
            admin_level=admin_level,
            top_k=top_k,
        )
        docs = self._apply_vector_boosts(raw_docs, intent)
        decision = self._decide_data_source(intent, docs)
        allowlist = self._select_allowed_tools(intent, docs, decision)
        
        # Manual 컬렉션 검색 (안전수칙, FAQ 등)
        manual_docs = self._search_manual_if_relevant(guarded_question, intent)
        
        context = self._build_rag_context(intent, docs, decision, manual_docs)
        return {
            "intent": intent,
            "decision": decision,
            "vector_docs": docs,
            "manual_docs": manual_docs,
            "tool_allowlist": allowlist,
            "context": context,
        }

    def route_only(
        self,
        question: str,
        user_id: int,
        admin_level: int | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        guarded_question = self._apply_guardrail_input(question)
        admin_level = self._resolve_admin_level(user_id, admin_level)
        intent = self._classify_intent(guarded_question)
        raw_docs = search_knowledge(
            query=guarded_question,
            user_id=user_id,
            admin_level=admin_level,
            top_k=top_k,
        )
        docs = self._apply_vector_boosts(raw_docs, intent)
        decision = self._decide_data_source(intent, docs)
        return {
            "intent": intent,
            "decision": decision,
            "vector_docs": docs,
        }

    def _resolve_admin_level(self, user_id: int, admin_level: int | None) -> int:
        if admin_level is not None:
            return int(admin_level)
        profile = get_user_profile(user_id) or {}
        return int(profile.get("admin_level", 0) or 0)

    def _classify_intent(self, question: str) -> dict[str, Any]:
        examples = [
            {
                "q": "내 대여 내역 좀 알려줘",
                "intent": "personal_data",
                "data_source": "mysql_only",
                "mysql_tables": ["rentals"],
                "why": "개인 대여 기록 조회",
            },
            {
                "q": "자전거 대여는 어떻게 하나요?",
                "intent": "general_knowledge",
                "data_source": "vector_only",
                "mysql_tables": [],
                "why": "일반 안내 문서로 충분",
            },
            {
                "q": "이번 달 결제 총액 알려줘",
                "intent": "personal_data",
                "data_source": "mysql_only",
                "mysql_tables": ["payments"],
                "why": "개인 결제 데이터 필요",
            },
            {
                "q": "지금 내 주변에 대여 가능한 자전거 있니?",
                "intent": "realtime_location",
                "data_source": "mysql_only",
                "mysql_tables": ["bikes"],
                "why": "실시간 위치 데이터 필요",
            },
            {
                "q": "환불 규정이 어떻게 돼?",
                "intent": "general_knowledge",
                "data_source": "vector_only",
                "mysql_tables": [],
                "why": "정책 문서 기반 답변",
            },
            {
                "q": "너가 할 수 있는 기능이 뭐야?",
                "intent": "general_knowledge",
                "data_source": "hybrid",
                "mysql_tables": [],
                "why": "기능 안내는 시스템 프롬프트 기반으로 응답",
            },
        ]

        examples_text = "\n".join(
            [
                (
                    f"- 질문: {ex['q']}\n"
                    f"  intent: {ex['intent']}\n"
                    f"  data_source: {ex['data_source']}\n"
                    f"  mysql_tables: {ex['mysql_tables']}\n"
                    f"  이유: {ex['why']}"
                )
                for ex in examples
            ]
        )

        system_prompt = (
            "너는 자전거 공유 서비스의 질문 분류기다. "
            "의도(intent), 데이터 소스(data_source), mysql_tables, confidence, reasoning을 JSON으로 출력한다. "
            "데이터 소스는 vector_only/mysql_only/hybrid 중 하나만 사용한다."
        )
        user_prompt = (
            f"다음 예시를 참고해 질문을 분류하라.\n\n"
            f"{examples_text}\n\n"
            f"질문: {question}\n\n"
            "다음 JSON 형식으로만 답변:\n"
            "{"
            '"intent":"...",'
            '"data_source":"...",'
            '"mysql_tables":[...],'
            '"confidence":0.0,'
            '"reasoning":"..."'
            "}"
        )

        response = self.llm_client.create_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[],
        )
        parsed = _safe_json_parse(response.content or "")
        return {
            "intent": parsed.get("intent", "general_knowledge"),
            "data_source": parsed.get("data_source", "hybrid"),
            "mysql_tables": parsed.get("mysql_tables", []),
            "confidence": float(parsed.get("confidence", 0.5) or 0.5),
            "reasoning": parsed.get("reasoning", ""),
        }

    def _apply_vector_boosts(
        self, raw_docs: list[dict[str, Any]], intent: dict[str, Any]
    ) -> list[KnowledgeDocument]:
        boosted: list[KnowledgeDocument] = []
        for doc in raw_docs:
            metadata = doc.get("metadata") or {}
            base_score = float(doc.get("score", 0.0) or 0.0)
            final_score = self._boost_score(base_score, metadata, intent)
            boosted.append(
                {
                    "content": doc.get("content", ""),
                    "metadata": metadata,
                    "score": base_score,
                    "final_score": final_score,
                }
            )
        boosted.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
        return boosted

    def _boost_score(
        self, base_score: float, metadata: KnowledgeMetadata, intent: dict[str, Any]
    ) -> float:
        if base_score <= 0:
            return 0.0
        intent_tags = metadata.get("intent_tags") or []
        importance = int(metadata.get("importance", 5) or 5)
        freshness = float(metadata.get("freshness_score", 1.0) or 1.0)
        feedback_score = float(metadata.get("feedback_score", 0.0) or 0.0)
        doc_type = str(metadata.get("doc_type", "") or "")
        created_at = metadata.get("created_at")

        intent_boost = 1.0 + (self.weights.intent_boost if intent.get("intent") in intent_tags else 0.0)
        importance_boost = 1.0 + (self.weights.importance_boost * ((importance - 5) / 5))
        freshness_boost = 1.0 + (self.weights.freshness_boost * (freshness - 0.5))
        feedback_boost = 1.0 + (self.weights.feedback_boost * max(-1.0, min(feedback_score, 1.0)))
        recency_factor = self._recency_factor(created_at)
        recency_boost = 1.0 + (self.weights.recency_boost * (recency_factor - 0.5))
        doc_type_boost = self._doc_type_weight(doc_type)

        return max(
            0.0,
            base_score
            * intent_boost
            * importance_boost
            * freshness_boost
            * feedback_boost
            * recency_boost
            * doc_type_boost,
        )

    def _decide_data_source(self, intent: dict[str, Any], docs: list[KnowledgeDocument]) -> dict[str, Any]:
        llm_confidence = float(intent.get("confidence", 0.5) or 0.5)
        llm_source = intent.get("data_source", "hybrid")

        top_score = float(docs[0].get("final_score", 0.0)) if docs else 0.0
        vector_confidence = 1 - math.exp(-top_score) if top_score > 0 else 0.0

        mysql_needed = bool(intent.get("mysql_tables")) or any(
            (doc.get("metadata") or {}).get("requires_mysql") is True for doc in docs
        )
        mysql_signal = 1.0 if mysql_needed else 0.0
        history_signal = self._history_mysql_signal(docs)

        combined = (
            llm_confidence * self.weights.llm_confidence
            + vector_confidence * self.weights.vector_confidence
            + mysql_signal * self.weights.mysql_signal
            + history_signal * self.weights.history_signal
        )

        if llm_source == "mysql_only" and llm_confidence >= self.weights.mysql_force_threshold:
            return {
                "data_source": "mysql_only",
                "confidence": llm_confidence,
                "reasoning": "LLM이 MySQL 필요로 판단",
            }

        if mysql_needed and combined >= self.weights.hybrid_threshold:
            return {
                "data_source": "hybrid" if llm_source == "hybrid" else "mysql_only",
                "confidence": combined,
                "reasoning": "MySQL 신호 및 결합 점수 고려",
            }

        if vector_confidence >= self.weights.vector_force_threshold and llm_source in {
            "vector_only",
            "hybrid",
        }:
            return {
                "data_source": "vector_only",
                "confidence": vector_confidence,
                "reasoning": "Vector 문서 신뢰도 우수",
            }

        final_source: DataSourceType = (
            llm_source if combined >= self.weights.hybrid_threshold else "hybrid"
        )
        return {
            "data_source": final_source,
            "confidence": combined,
            "reasoning": "LLM/Vector 결합 점수 기준",
        }

    def _fetch_mysql_data(
        self,
        intent: dict[str, Any],
        user_id: int,
        admin_level: int,
        decision: dict[str, Any],
    ) -> dict[str, Any] | None:
        if decision.get("data_source") not in {"mysql_only", "hybrid"}:
            return None

        tables = [str(name) for name in intent.get("mysql_tables", []) if name]
        data: dict[str, Any] = {}

        if admin_level < 1 and intent.get("intent") == "personal_data":
            # 일반 사용자는 본인 데이터만 조회하도록 유지
            pass

        if "rentals" in tables:
            data["rentals"] = get_rentals(user_id=user_id, days=30)
            data["usage_summary"] = get_total_usage(user_id=user_id)

        if "payments" in tables:
            data["payments"] = get_payments(user_id=user_id, limit=20)
            data["total_payments"] = get_total_payments(user_id=user_id)

        if "bikes" in tables:
            data["bikes"] = get_available_bikes()

        if "users" in tables:
            data["user_profile"] = get_user_profile(user_id=user_id)

        if "inquiries" in tables:
            data["inquiries"] = get_inquiries(user_id=user_id)

        return data or None

    def _generate_answer(
        self,
        question: str,
        intent: dict[str, Any],
        docs: list[KnowledgeDocument],
        mysql_data: dict[str, Any] | None,
    ) -> str:
        context_parts: list[str] = []
        doc_limit_raw = os.getenv("RAG_CONTEXT_DOCS", "2").strip()
        snippet_limit_raw = os.getenv("RAG_CONTEXT_SNIPPET_CHARS", "200").strip()
        try:
            doc_limit = max(1, min(int(doc_limit_raw), 5))
        except ValueError:
            doc_limit = 2
        try:
            snippet_limit = max(80, min(int(snippet_limit_raw), 500))
        except ValueError:
            snippet_limit = 200

        if docs:
            context_parts.append("=== 관련 문서 ===")
            for idx, doc in enumerate(docs[:doc_limit], 1):
                snippet = (doc.get("content") or "").strip()
                if len(snippet) > snippet_limit:
                    snippet = snippet[:snippet_limit] + "..."
                context_parts.append(f"{idx}. {snippet}")
            context_parts.append("")

        if mysql_data:
            context_parts.append("=== 사용자 데이터 ===")
            context_parts.append(json.dumps(mysql_data, ensure_ascii=False))
            context_parts.append("")

        context = "\n".join(context_parts)
        system_prompt = (
            "너는 근거 기반 답변 생성기다. "
            "컨텍스트가 부족하면 일반적으로 알려진 안전한 범위의 설명을 제공하고, "
            "확인이 필요한 부분은 확인이 필요하다고 밝혀라."
        )
        user_prompt = (
            f"질문: {question}\n"
            f"의도: {intent.get('intent')}\n\n"
            f"{context}\n"
            "답변 규칙:\n"
            "1. 컨텍스트가 있으면 그 내용을 우선 사용\n"
            "2. 컨텍스트가 없으면 일반적인 안내 수준으로 답변\n"
            "3. 숫자는 정확히 유지\n"
            "4. 한국어로 간결하게 작성\n"
        )
        response = self.llm_client.create_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[],
        )
        answer = (response.content or "").strip()
        return self._apply_guardrail_output(answer)

    def _store_conversation(
        self, user_id: int, question: str, answer: str, intent: dict[str, Any]
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
            # 저장 실패는 응답 생성에 영향 주지 않음
            logger.exception("MongoDB 대화 저장 실패")
            return

    def _select_allowed_tools(
        self,
        intent: dict[str, Any],
        docs: list[KnowledgeDocument],
        decision: dict[str, Any],
    ) -> list[str]:
        if decision.get("data_source") == "vector_only":
            return []

        tables = {str(name) for name in intent.get("mysql_tables", []) if name}
        for doc in docs:
            metadata = doc.get("metadata") or {}
            if metadata.get("requires_mysql") is True:
                for table in metadata.get("mysql_tables", []) or []:
                    tables.add(str(table))

        table_to_tools = {
            "rentals": ["get_rentals", "get_usage_summary", "get_total_usage"],
            "payments": ["get_payments", "get_pricing_summary", "get_total_payments"],
            "bikes": ["get_available_bikes", "get_nearby_stations"],
            "users": ["get_user_profile"],
            "inquiries": ["get_inquiries"],
        }

        allowed: list[str] = []
        for table in tables:
            allowed.extend(table_to_tools.get(table, []))

        if intent.get("intent") == "personal_data":
            allowed.append("get_user_profile")

        deduped: list[str] = []
        seen: set[str] = set()
        for name in allowed:
            if name not in seen:
                seen.add(name)
                deduped.append(name)
        return deduped

    def _search_manual_if_relevant(
        self,
        question: str,
        intent: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """안전수칙, FAQ 관련 질문이면 manual 컬렉션 검색"""
        # 키워드 기반 검색 트리거
        manual_keywords = [
            "안전", "수칙", "주의", "헬멧", "규칙", "규정",
            "어떻게", "방법", "문의", "FAQ", "질문",
            "결제", "충전", "이용", "사용법",
        ]
        question_lower = question.lower()
        
        # 키워드 매칭 또는 general_knowledge 의도일 때 검색
        should_search = (
            intent.get("intent") == "general_knowledge"
            or any(kw in question for kw in manual_keywords)
        )
        
        if not should_search:
            return []
        
        return search_manual_knowledge(query=question, top_k=3)

    def _build_rag_context(
        self,
        intent: dict[str, Any],
        docs: list[KnowledgeDocument],
        decision: dict[str, Any],
        manual_docs: list[dict[str, Any]] | None = None,
    ) -> str:
        parts: list[str] = [
            f"RAG 판단: data_source={decision.get('data_source')} "
            f"confidence={decision.get('confidence'):.2f}"
        ]
        
        # Manual 컬렉션 결과 (안전수칙, FAQ) - 최우선 표시
        if manual_docs:
            manual_context = format_manual_results_as_context(manual_docs)
            if manual_context:
                parts.append(manual_context)
        
        history_docs = [
            doc
            for doc in docs
            if (doc.get("metadata") or {}).get("doc_type")
            in {"conversation", "assistant_reply"}
        ]
        if history_docs:
            parts.append("최근 대화 힌트:")
            for doc in history_docs[:2]:
                snippet = (doc.get("content") or "").strip()
                if len(snippet) > 120:
                    snippet = snippet[:120] + "..."
                parts.append(f"- {snippet}")

        if docs:
            parts.append("관련 문서 요약:")
            for doc in docs[:2]:
                snippet = (doc.get("content") or "").strip()
                if len(snippet) > 120:
                    snippet = snippet[:120] + "..."
                parts.append(f"- {snippet}")

        if intent:
            parts.append(
                f"의도={intent.get('intent')} data_source={intent.get('data_source')} "
                f"mysql_tables={intent.get('mysql_tables')}"
            )
        return "\n".join(parts)

    def _history_mysql_signal(self, docs: list[KnowledgeDocument]) -> float:
        for doc in docs:
            metadata = doc.get("metadata") or {}
            if metadata.get("doc_type") not in {"conversation", "assistant_reply"}:
                continue
            tags = set()
            for item in metadata.get("intent_tags") or []:
                if isinstance(item, (str, int, float)):
                    tags.add(str(item))
            if tags.intersection({"personal_data", "realtime_location", "transaction", "history"}):
                return 1.0
        return 0.0

    def _doc_type_weight(self, doc_type: str) -> float:
        if doc_type == "user_profile_memory":
            return 1.35
        if doc_type == "assistant_reply":
            return 1.15
        if doc_type == "conversation":
            return 1.05
        return 1.0

    def _recency_factor(self, created_at: Any) -> float:
        dt = self._parse_datetime(created_at)
        if not dt:
            return 0.5
        now = datetime.now(tz=timezone.utc)
        age_seconds = max(0.0, (now - dt).total_seconds())
        # 약 7일 시점에 0.5 정도로 완만히 감소
        half_life = 7 * 24 * 60 * 60
        return 0.5 ** (age_seconds / half_life)

    def _parse_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _apply_guardrail_input(self, text: str) -> str:
        if not self.guardrail_client:
            return text
        decision = self.guardrail_client.apply(text=text, source="INPUT")
        cleaned = (decision.output_text or "").strip()
        return cleaned or text

    def _apply_guardrail_output(self, text: str) -> str:
        if not self.guardrail_client:
            return text
        decision = self.guardrail_client.apply(text=text, source="OUTPUT")
        cleaned = (decision.output_text or "").strip()
        return cleaned or text


def _safe_json_parse(text: str) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
