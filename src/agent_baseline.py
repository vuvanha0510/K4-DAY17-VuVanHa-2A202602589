from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None
        if not force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        # TODO: return cumulative agent token count for one thread.
        return self._session_for(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        # TODO: estimate how much prompt context this baseline kept processing.
        return self._session_for(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self._session_for(thread_id)

        # 1. Prompt cost: the baseline re-sends the whole thread every turn.
        context_tokens = estimate_tokens(self._render(session))
        session.prompt_tokens_processed += context_tokens

        # 2. The user turn joins this thread only.
        session.messages.append({"role": "user", "content": message})

        # 3. A deterministic answer that may only use *this* thread.
        answer = self._offline_response(session, message)
        session.messages.append({"role": "assistant", "content": answer})

        # 4. Only what the agent generates counts as "agent tokens".
        generated = estimate_tokens(answer)
        session.token_usage += generated

        return {
            "answer": answer,
            "agent_tokens": generated,
            "prompt_tokens": context_tokens,
            "thread_id": thread_id,
            "mode": "offline",
        }

    def _offline_response(self, session: SessionState, message: str) -> str:
        """Answer from the current thread only.

        Anything the user said in another thread is invisible here, which is the
        whole point of the baseline: same thread = remembered, new thread = gone.
        """

        hits = self._search_thread(session, message)
        if hits:
            return "Trong thread này mình nhớ: " + " ".join(f"- {h}" for h in hits)

        turns = sum(1 for m in session.messages if m["role"] == "user")
        return (
            f"Mình đã ghi nhận lượt này (thread này có {turns} lượt của bạn). "
            "Mình chỉ nhớ được nội dung của thread hiện tại, nên sang thread mới "
            "các thông tin đã nói sẽ không còn."
        )

    @staticmethod
    def _search_thread(session: SessionState, message: str, limit: int = 2) -> list[str]:
        """Find sentences already said *in this thread* that match the question."""

        keywords = [
            word.lower()
            for word in re.findall(r"[\wÀ-ỹ]{3,}", message or "")
            if word.lower() not in _STOP_WORDS
        ]
        if not keywords:
            return []

        hits: list[str] = []
        for entry in session.messages:
            if entry["role"] != "user":
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", entry["content"]):
                lowered = sentence.lower()
                if any(keyword in lowered for keyword in keywords):
                    text = sentence.strip()
                    if text and text not in hits:
                        hits.append(text)
                    break
            if len(hits) >= limit:
                break
        return hits

    def _session_for(self, thread_id: str) -> SessionState:
        """Get (or create) the short-term memory of one thread."""

        session = self.sessions.get(thread_id)
        if session is None:
            session = SessionState()
            self.sessions[thread_id] = session
        return session

    @staticmethod
    def _render(session: SessionState) -> str:
        """Flatten a session into the text the model would receive."""

        return "\n".join(
            f"{message['role']}: {message['content']}" for message in session.messages
        )

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live path, kept thin so the memory logic stays testable offline."""

        result = self.langchain_agent.invoke(  # type: ignore[union-attr]
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        text = _extract_text(result)

        session = self._session_for(thread_id)
        context_tokens = estimate_tokens(self._render(session))
        session.prompt_tokens_processed += context_tokens
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": text})

        generated = estimate_tokens(text)
        session.token_usage += generated
        return {
            "answer": text,
            "agent_tokens": generated,
            "prompt_tokens": context_tokens,
            "thread_id": thread_id,
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.

        Any missing piece (no LangChain installed, no API key, no base URL for a
        custom provider) simply returns None, and the lab falls back to the
        deterministic offline path.
        """

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None

        try:
            model = build_chat_model(self.config.model)
        except Exception:
            # No credentials / unsupported provider -> stay offline.
            return None

        return create_agent(model, tools=[], checkpointer=InMemorySaver())


def _extract_text(result: Any) -> str:
    """Pull the text out of a LangChain agent result."""

    messages: Any = None
    if isinstance(result, dict):
        messages = result.get("messages")
    if messages is None:
        messages = getattr(result, "messages", None)

    for message in reversed(list(messages or [])):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):  # content blocks
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if parts:
                return "\n".join(parts)
    return str(result)


# Question words that carry no retrieval signal.
_STOP_WORDS = frozenset(
    {
        "của",
        "cho",
        "với",
        "những",
        "người",
        "mình",
        "bạn",
        "lượt",
        "giờ",
        "làm",
        "nào",
        "gì",
        "đây",
        "đó",
        "nhớ",
        "nhắc",
        "lại",
        "giúp",
        "biết",
        "thử",
        "về",
    }
)
