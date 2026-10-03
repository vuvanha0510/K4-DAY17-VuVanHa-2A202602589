from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    SINGLE_VALUE_FACTS,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


# How a question maps to the facts that should answer it. Order matters only
# for readability: every matching group is answered, not just the first one.
_FACT_ROUTING: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("name", ("tên",), "Tên"),
    ("location", ("ở đâu", "nơi ở", "đang ở", "sống ở", "hiện ở"), "Nơi ở hiện tại"),
    ("profession", ("nghề", "việc làm", "công việc", "làm gì", "chức danh"), "Nghề nghiệp hiện tại"),
    ("favorite_drink", ("đồ uống", "uống", "nước"), "Đồ uống yêu thích"),
    ("favorite_food", ("món ăn", "món", "ăn"), "Món ăn yêu thích"),
    ("pet", ("nuôi", "con gì", "thú cưng", "pet", "giống"), "Con vật nuôi"),
    (
        "response_style",
        ("style", "trả lời", "cách trả lời", "kiểu trả lời", "bullet"),
        "Style trả lời",
    ),
    (
        "interests",
        ("quan tâm", "mối quan tâm", "thích", "học", "chủ đề", "kỹ thuật"),
        "Mối quan tâm",
    ),
)

_FACT_ORDER = tuple(key for key, _, _ in _FACT_ROUTING)


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None
        if not force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        # 1. + 2. Persistent memory: only stable facts, corrections win.
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            mode = "replace" if key in SINGLE_VALUE_FACTS else "union"
            self.profile_store.upsert_fact(user_id, key, value, mode=mode)

        # 3. Short-term memory with automatic compaction.
        self.compact_memory.append(thread_id, "user", message)

        # 4. Prompt cost: User.md + compact summary + kept messages.
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 5. Answer from the persisted profile, not from this thread alone.
        answer = self._offline_response(user_id, thread_id, message)

        # 6. Close the turn and account for generated tokens.
        self.compact_memory.append(thread_id, "assistant", answer)
        generated = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + generated

        return {
            "answer": answer,
            "agent_tokens": generated,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "mode": "offline",
            "memory_path": str(self.profile_store.path_for(user_id)),
            "memory_bytes": self.profile_store.file_size(user_id),
            "compactions": self.compaction_count(thread_id),
            "profile_updates": updates,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        return estimate_tokens(
            "\n".join(
                [
                    self.profile_store.read_text(user_id),
                    self.compact_memory.render(thread_id),
                ]
            )
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        The question is matched against the stored facts, so the answer only
        repeats what is relevant, and it still works in a brand new thread
        because the facts come from `User.md`, not from the thread history.
        """

        facts = self.profile_store.facts(user_id)
        if not facts:
            return (
                "Mình chưa lưu được thông tin ổn định nào về bạn. "
                "Bạn nói tên, nơi ở hoặc nghề nghiệp thì mình sẽ nhớ lại ở lượt sau."
            )

        selected = self._select_facts(facts, message)
        if not selected:
            selected = [(key, label) for key, label, _ in _FACT_ROUTING if facts.get(key)]
        if not selected:
            return "Mình có hồ sơ của bạn nhưng chưa trích được fact nào để trả lời."

        lines = [f"- {label}: {facts[key]}" for key, label in selected]
        answer = "Mình đọc từ User.md (memory bền vững):\n" + "\n".join(lines)
        if facts.get("response_style"):
            # Respect the preference the user already gave us.
            answer += "\n(Mình trả lời theo style đã lưu của bạn.)"
        return answer

    @staticmethod
    def _select_facts(facts: dict[str, str], message: str) -> list[tuple[str, str]]:
        """Pick the facts a question is actually about."""

        text = (message or "").lower()
        selected: list[tuple[str, str]] = []
        for key, triggers, label in _FACT_ROUTING:
            if not facts.get(key):
                continue
            if any(trigger in text for trigger in triggers):
                selected.append((key, label))
        return selected

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live path: the same three memory layers, driven by the real model."""

        updates = extract_profile_updates(message)
        for key, value in updates.items():
            mode = "replace" if key in SINGLE_VALUE_FACTS else "union"
            self.profile_store.upsert_fact(user_id, key, value, mode=mode)
        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        result = self.langchain_agent.invoke(  # type: ignore[union-attr]
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id, "user_id": user_id}},
        )
        text = _extract_text(result)
        self.compact_memory.append(thread_id, "assistant", text)

        generated = estimate_tokens(text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + generated
        return {
            "answer": text,
            "agent_tokens": generated,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "mode": "live",
            "memory_path": str(self.profile_store.path_for(user_id)),
            "memory_bytes": self.profile_store.file_size(user_id),
            "compactions": self.compaction_count(thread_id),
            "profile_updates": updates,
        }

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
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

        store = self.profile_store
        tools = [_read_profile_tool(store), _write_profile_tool(store)]
        return create_agent(model, tools=tools, checkpointer=InMemorySaver())


def _read_profile_tool(store: UserProfileStore):
    """Tool: let the live agent read the persistent `User.md`."""

    def read_user_profile(user_id: str) -> str:
        """Read the persistent profile of a user."""

        return store.read_text(user_id)

    return read_user_profile


def _write_profile_tool(store: UserProfileStore):
    """Tool: let the live agent update the persistent `User.md`."""

    def write_user_profile(user_id: str, key: str, value: str, mode: str = "union") -> str:
        """Store one profile fact such as `name`, `location` or `profession`."""

        store.upsert_fact(user_id, key, value, mode=mode)
        return store.read_text(user_id)

    return write_user_profile


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
