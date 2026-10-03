from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore

# A user statement that carries several stable facts at once.
PROFILE_TURN = (
    "Chào bạn, mình tên là DũngCT. Mình ở Đà Nẵng và đang làm backend engineer "
    "cho một startup AI. Đồ uống yêu thích của mình là cà phê sữa đá. "
    "Mình muốn bạn trả lời ngắn gọn, rõ ý."
)


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests.

    - `state_dir` points into `tmp_path`, so tests never touch the repo state.
    - A tiny compact threshold makes compaction happen within a few turns.
    - The model config is a placeholder: every test runs `force_offline=True`.
    """

    base = load_config(Path(__file__).resolve().parent.parent)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=base.base_dir,
        data_dir=base.data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=120,
        compact_keep_messages=2,
        model=base.model,
        judge_model=base.judge_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    user_id = "dungct"

    # 1. Read before anything exists -> empty default template.
    assert "chưa có dữ liệu" in store.read_text(user_id)

    # 2. Write creates the file at the documented path.
    path = store.write_text(user_id, "# User Profile: dungct\n\n## Profile\n- name: DũngCT\n")
    assert path.is_file()
    assert path.name == "User.md"
    assert store.file_size(user_id) > 0

    # 3. Facts are parsed back out of the markdown.
    assert store.facts(user_id) == {"name": "DũngCT"}

    # 4. Edit replaces exactly one occurrence.
    assert store.edit_text(user_id, "name: DũngCT", "name: Dũng") is True
    assert store.facts(user_id) == {"name": "Dũng"}
    # A search string that is absent reports no change.
    assert store.edit_text(user_id, "không có ở đây", "x") is False

    # 5. upsert_fact overwrites a corrected value.
    assert store.upsert_fact(user_id, "location", "Huế", mode="replace") is True
    assert store.facts(user_id)["location"] == "Huế"
    # Writing the same value again is a no-op.
    assert store.upsert_fact(user_id, "location", "Huế", mode="replace") is False

    # 6. union mode merges items and never duplicates them.
    store.upsert_fact(user_id, "response_style", "ngắn gọn", mode="union")
    store.upsert_fact(user_id, "response_style", "3 bullet", mode="union")
    store.upsert_fact(user_id, "response_style", "ngắn gọn", mode="union")
    assert store.facts(user_id)["response_style"] == "ngắn gọn; 3 bullet"


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""

    config = make_config(tmp_path)
    manager = CompactMemoryManager(
        threshold_tokens=config.compact_threshold_tokens,
        keep_messages=config.compact_keep_messages,
    )

    # A short thread stays untouched.
    manager.append("t1", "user", "Mình tên là DũngCT.")
    assert manager.compaction_count("t1") == 0
    assert manager.thread_tokens("t1") <= config.compact_threshold_tokens

    # Long turns push the live context over the threshold.
    for index in range(8):
        manager.append("t1", "user", f"Lượt {index}: " + "ngữ cảnh rất dài " * 20)
        manager.append("t1", "assistant", f"Trả lời {index}: " + "chi tiết " * 20)

    assert manager.compaction_count("t1") > 0

    context = manager.context("t1")
    assert context["compactions"] == manager.compaction_count("t1")
    # `total_messages` counts every append, even the ones already compacted away.
    assert context["total_messages"] == 17
    assert len(context["messages"]) < context["total_messages"]
    # Only the recent messages are kept in full; the rest moved into a summary.
    assert len(context["messages"]) <= config.compact_keep_messages
    assert context["summary"]

    # Threads are independent.
    assert manager.compaction_count("t2") == 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)
    user_id = "dungct"

    # Teach both agents the same facts in thread 1.
    advanced.reply(user_id, "thread-1", PROFILE_TURN)
    baseline.reply(user_id, "thread-1", PROFILE_TURN)

    # 1. Inside the same thread both can quote the facts back.
    same_thread = advanced.reply(user_id, "thread-1", "Mình tên gì?")
    assert "DũngCT" in same_thread["answer"]

    # 2. A brand-new thread: only persistent memory can answer.
    advanced_answer = advanced.reply(user_id, "thread-2", "Mình tên gì?")["answer"]
    baseline_answer = baseline.reply(user_id, "thread-2", "Mình tên gì?")["answer"]

    assert "DũngCT" in advanced_answer
    assert "DũngCT" not in baseline_answer

    # 3. The advanced agent also answers other routed facts across sessions.
    for question, expected in (
        ("Hiện tại mình đang ở đâu?", "Đà Nẵng"),
        ("Mình làm nghề gì?", "backend engineer"),
        ("Đồ uống mình thích là gì?", "cà phê sữa đá"),
    ):
        assert expected in advanced.reply(user_id, "thread-3", question)["answer"]

    # 4. Persistent memory really is a file on disk.
    assert advanced.memory_file_size(user_id) > 0
    assert not hasattr(baseline, "memory_file_size")

    # 5. A correction must win over the fact it replaces.
    advanced.reply(user_id, "thread-4", "Mình vừa chuyển từ Đà Nẵng sang Huế, giờ đang ở Huế.")
    assert "Huế" in advanced.reply(user_id, "thread-5", "Mình đang ở đâu?")["answer"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)
    user_id = "dungct"

    long_turns = [
        "Mình tên là DũngCT và đang làm backend engineer cho một startup AI ở Đà Nẵng. "
        "Bài báo hôm nay nói về " + "chi tiết rất dài " * 25,
        *("Mình lại đọc một đoạn tin nữa rất dài. " + "nội dung tin dài " * 25 for _ in range(6)),
    ]

    for turn in long_turns:
        advanced.reply(user_id, "long-thread", turn)
        baseline.reply(user_id, "long-thread", turn)

    advanced_prompt = advanced.prompt_token_usage("long-thread")
    baseline_prompt = baseline.prompt_token_usage("long-thread")

    # Compact memory keeps the advanced agent's live context bounded, while the
    # baseline re-sends the entire (growing) thread on every single turn.
    assert advanced.compaction_count("long-thread") > 0
    assert advanced_prompt < baseline_prompt
    # The baseline never compacts, so its cost grows with the thread.
    assert baseline.compaction_count("long-thread") == 0
    # Both agents still generated output.
    assert advanced.token_usage("long-thread") > 0
    assert baseline.token_usage("long-thread") > 0
