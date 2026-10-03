from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import estimate_tokens


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


_HEADERS = (
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
)


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk.

    The file is a list of objects shaped like::

        {"id", "user_id", "turns": [...], "recall_questions": [...]}

    A single object (or an object wrapping the list under `conversations`) is
    accepted too, so the loader tolerates small dataset edits.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("conversations", [payload])
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected dataset format in {file_path}: expected a list.")

    conversations: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        turns = [str(turn) for turn in item.get("turns", []) if str(turn).strip()]
        if not turns:
            continue
        conversations.append(
            {
                "id": str(item.get("id") or f"conv-{index + 1:02d}"),
                "user_id": str(item.get("user_id") or "anonymous"),
                "turns": turns,
                "recall_questions": [
                    question
                    for question in item.get("recall_questions", [])
                    if isinstance(question, dict)
                ],
            }
        )
    return conversations


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace so matching ignores formatting."""

    return " ".join((text or "").lower().split())


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""

    wanted = [str(item) for item in (expected or []) if str(item).strip()]
    if not wanted:
        # Nothing to check: a non-empty answer is the best we can ask for.
        return 1.0 if (answer or "").strip() else 0.0

    haystack = _normalize(answer)
    hits = sum(1 for item in wanted if _normalize(item) in haystack)
    return round(hits / len(wanted), 4)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight quality score for offline mode.

    Two cheap, deterministic signals, because no real judge model is available
    offline:
    1. *Correctness*: the share of `expected` strings the answer contains.
    2. *Economy*: a mild bonus for staying concise, which is what the stored
       `response_style` preference asks for; long answers are penalized.

    The result is normalized to `0.0 - 1.0`.
    """

    text = (answer or "").strip()
    if not text:
        return 0.0

    correctness = recall_points(text, expected)
    # 120 tokens is "comfortably short" for a recall answer.
    length_tokens = max(1, estimate_tokens(text))
    economy = 1.0 if length_tokens <= 120 else max(0.3, 120 / length_tokens)

    return round(0.7 * correctness + 0.3 * economy, 4)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    1. Feed all turns of a conversation into its own thread.
    2. Track `agent tokens only` (what the agent generated).
    3. Track `prompt tokens processed` (the context it had to drag along).
    4. Ask the recall questions in a *fresh* thread, so only long-term memory
       can answer them.
    5. Average recall and quality over all questions.
    6. Record memory file growth and the compaction count.
    """

    agent_tokens = 0
    prompt_tokens = 0
    compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    user_ids: list[str] = []

    for conversation in conversations:
        conv_id = str(conversation["id"])
        user_id = str(conversation["user_id"])
        thread_id = conv_id
        # A brand-new thread id: nothing from the conversation may leak here.
        recall_thread = f"{conv_id}::recall"
        if user_id not in user_ids:
            user_ids.append(user_id)

        # 1-3. Play the whole conversation.
        for turn in conversation["turns"]:
            result = agent.reply(user_id, thread_id, turn)
            agent_tokens += int(result.get("agent_tokens", 0))
            prompt_tokens += int(result.get("prompt_tokens", 0))

        compactions += int(agent.compaction_count(thread_id))

        # 4-5. Cross-session recall questions, asked in a fresh thread.
        for question in conversation["recall_questions"]:
            text = str(question.get("question", ""))
            expected = [str(item) for item in question.get("expected_contains", [])]
            answer = str(agent.reply(user_id, recall_thread, text).get("answer", ""))
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    # 6. Persistent memory footprint (the baseline has none).
    memory_bytes = 0
    for user_id in user_ids:
        size_fn = getattr(agent, "memory_file_size", None)
        if callable(size_fn):
            memory_bytes += int(size_fn(user_id))

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=(
            round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0
        ),
        response_quality=(
            round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0.0
        ),
        memory_growth_bytes=memory_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render benchmark rows as a table (tabulate when available)."""

    table = [
        [
            row.agent_name,
            f"{row.agent_tokens_only:,}",
            f"{row.prompt_tokens_processed:,}",
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            f"{row.memory_growth_bytes:,}",
            str(row.compactions),
        ]
        for row in rows
    ]

    try:
        from tabulate import tabulate

        return tabulate(table, headers=list(_HEADERS), tablefmt="github")
    except ImportError:
        pass

    # Markdown fallback so the lab still runs without `tabulate`.
    widths = [len(header) for header in _HEADERS]
    for line in table:
        for index, cell in enumerate(line):
            widths[index] = max(widths[index], len(cell))
    header = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(_HEADERS)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)) + " |"
        for line in table
    ]
    return "\n".join([header, separator, *body])


def _reset_profiles(config) -> None:
    """Start every benchmark run from an empty persistent memory."""

    profiles = Path(config.profiles_dir)
    if not profiles.is_dir():
        return
    for user_dir in profiles.iterdir():
        if user_dir.is_dir():
            shutil.rmtree(user_dir, ignore_errors=True)


def main() -> None:
    """Run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions

    Both agents run with `force_offline=True`: the benchmark must be
    deterministic and runnable without any API key, since what is measured
    here is the memory system, not the provider.
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    suites = (
        ("Standard Benchmark", config.standard_dataset),
        ("Long-Context Stress Benchmark", config.stress_dataset),
    )

    for title, dataset_path in suites:
        conversations = load_conversations(dataset_path)
        _reset_profiles(config)

        print(f"\n=== {title} ===")
        print(f"dataset: {dataset_path}")
        print(
            f"conversations: {len(conversations)} | "
            f"turns: {sum(len(c['turns']) for c in conversations)} | "
            f"recall questions: {sum(len(c['recall_questions']) for c in conversations)}"
        )

        rows = [
            # `force_offline=True` keeps the benchmark deterministic and free:
            # the point being measured is the memory system, not the provider.
            run_agent_benchmark(
                "Baseline",
                BaselineAgent(config=config, force_offline=True),
                conversations,
                config,
            ),
            run_agent_benchmark(
                "Advanced",
                AdvancedAgent(config=config, force_offline=True),
                conversations,
                config,
            ),
        ]

        print(format_rows(rows))
        _print_analysis(rows)

    print("\nGợi ý đọc kết quả:")
    print(
        "  - Hội thoại ngắn: Advanced có thể tốn token hơn Baseline vì phải "
        "đọc cả User.md mỗi lượt."
    )
    print(
        "  - Hội thoại dài: compact memory giữ prompt tokens của Advanced thấp "
        "hơn hẳn Baseline, đổi lại lịch sử cũ bị nén (lossy)."
    )
    print(
        "  - Baseline có cross-session recall = 0 vì không có persistent memory."
    )


def _print_analysis(rows: list[BenchmarkRow]) -> None:
    """Print the delta between the two agents, which is the actual lesson."""

    by_name = {row.agent_name: row for row in rows}
    baseline = by_name.get("Baseline")
    advanced = by_name.get("Advanced")
    if baseline is None or advanced is None:
        return

    def delta(field: str) -> str:
        left = getattr(baseline, field)
        right = getattr(advanced, field)
        if isinstance(left, float) or isinstance(right, float):
            # Avoid float noise such as 0.8631000000000001 in the report.
            value = round(float(right) - float(left), 4)
            return f"{value:+.2f}"
        sign = "+" if right - left > 0 else ""
        return f"{sign}{right - left}"

    print(
        f"Advanced - Baseline | agent tokens: {delta('agent_tokens_only')} | "
        f"prompt tokens: {delta('prompt_tokens_processed')} | "
        f"recall: {delta('recall_score')} | "
        f"quality: {delta('response_quality')}"
    )


if __name__ == "__main__":
    main()
