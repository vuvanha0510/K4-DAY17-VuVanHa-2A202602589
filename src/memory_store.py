from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Average characters per token. Vietnamese syllables are long, so a plain
# English 4-chars-per-token rule is too low; 3.5 tracks real tokenizers
# closely enough for an offline benchmark and stays perfectly deterministic.
CHARS_PER_TOKEN = 3.5

# Section of `User.md` that holds the machine-managed facts.
FACTS_HEADING = "## Profile"

# Canonical fact order used when rendering `User.md`.
FACT_ORDER: tuple[str, ...] = (
    "name",
    "location",
    "profession",
    "response_style",
    "favorite_drink",
    "favorite_food",
    "pet",
    "interests",
)


def estimate_tokens(text: str) -> int:
    """Simple, deterministic token estimator.

    A real tokenizer is not required for this lab: the benchmark only needs a
    stable approximation of "how much context does the agent carry".
    """

    if not text:
        return 0
    stripped = str(text).strip()
    if not stripped:
        return 0
    return max(1, int(round(len(stripped) / CHARS_PER_TOKEN)))


def default_profile_markdown(user_id: str) -> str:
    """Empty `User.md` template, used when a profile does not exist yet."""

    return f"# User Profile: {user_id}\n\n{FACTS_HEADING}\n- (chưa có dữ liệu)\n"


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    # ------------------------------------------------------------------ paths
    def path_for(self, user_id: str) -> Path:
        """Return `User.md` for a user, creating its folder on demand."""

        folder = self.root_dir / self._slugify(user_id)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "User.md"

    @staticmethod
    def _slugify(user_id: str) -> str:
        """Sanitize a user id so it is safe to use as a folder name."""

        raw = (user_id or "anonymous").strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
        return slug or "anonymous"

    # ------------------------------------------------------------- read/write
    def read_text(self, user_id: str) -> str:
        """Return the file content, or an empty default profile."""

        path = self.path_for(user_id)
        if not path.is_file():
            return default_profile_markdown(user_id)
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown to disk and return the file path."""

        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence inside `User.md`; return whether it changed."""

        path = self.path_for(user_id)
        if not path.is_file() or search_text not in path.read_text(encoding="utf-8"):
            return False
        content = path.read_text(encoding="utf-8")
        path.write_text(
            content.replace(search_text, replacement, 1), encoding="utf-8"
        )
        return True

    def file_size(self, user_id: str) -> int:
        """Return the current file size in bytes (0 when missing)."""

        path = self.path_for(user_id)
        return path.stat().st_size if path.is_file() else 0

    # ------------------------------------------------------------------ facts
    def facts(self, user_id: str) -> dict[str, str]:
        """Parse the `- key: value` lines out of `User.md`."""

        return self.parse_facts(self.read_text(user_id))

    @staticmethod
    def parse_facts(content: str) -> dict[str, str]:
        """Extract the managed facts from markdown text."""

        facts: dict[str, str] = {}
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue
            body = stripped.lstrip("-").strip()
            if ":" not in body:
                continue
            key, _, value = body.partition(":")
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if not key or not value or value.startswith("("):
                continue
            facts[key] = value
        return facts

    @staticmethod
    def render_facts(facts: dict[str, str], user_id: str) -> str:
        """Render facts back into the `User.md` template."""

        keys = [k for k in FACT_ORDER if facts.get(k)]
        keys += [k for k in sorted(facts) if k not in FACT_ORDER and facts.get(k)]
        lines = [f"# User Profile: {user_id}", "", FACTS_HEADING]
        lines += [f"- {key}: {facts[key]}" for key in keys]
        if len(lines) == 3:
            lines.append("- (chưa có dữ liệu)")
        return "\n".join(lines) + "\n"

    def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        mode: str = "replace",
        max_items: int = 12,
    ) -> bool:
        """Insert or update one fact in `User.md`.

        `mode="replace"` overwrites the value, which is what a correction needs
        ("không còn ở Đà Nẵng nữa, giờ ở Huế"). `mode="union"` merges the value
        item by item, which is what a stable preference needs ("ngắn gọn"
        together with "3 bullet").

        Merging happens per item, so the same preference repeated in ten turns
        is stored once, and `max_items` keeps a multi-value fact from growing
        without bound.

        Returns True when the file actually changed.
        """

        key = key.strip().lower().replace(" ", "_")
        value = _clean_value(value)
        if not key or not value:
            return False

        facts = self.facts(user_id)

        if mode == "union":
            merged: list[str] = []
            seen: set[str] = set()
            # Current items first, so the earliest phrasing is preserved.
            for part in _split_items(str(facts.get(key, ""))) + _split_items(value):
                marker = part.lower()
                if marker not in seen:
                    seen.add(marker)
                    merged.append(part)
            value = "; ".join(merged[-max_items:])

        if facts.get(key) == value:
            return False

        facts[key] = value
        self.write_text(user_id, self.render_facts(facts, user_id))
        return True


# Connectives that must never end up inside a stored fact value.
_TRAILING_NOISE = (
    "như cũ",
    "như trước",
    "nhưng",
    "như",
    "vì",
    "cho",
    "để",
    "rồi",
    "thôi",
    "ạ",
    "nhé",
    "nha",
    "đó",
    "thế",
)

# Words that mean "the user is asking", not "the user is telling".
_QUESTION_WORDS = (
    "gì",
    "bao nhiêu",
    "thế nào",
    "ai",
    "nào",
    "có phải",
    "thử",
    "xem có",
)


def _clean_value(value: str) -> str:
    """Trim punctuation and dangling connectives off an extracted value."""

    text = re.sub(r"\s+", " ", (value or "").strip())
    text = text.strip(" \t\r\n.,;:!?-–—\"'“”‘’()[]")
    # Drop trailing connectives ("cà phê sữa đá như cũ" -> "cà phê sữa đá").
    changed = True
    while changed and text:
        changed = False
        for noise in _TRAILING_NOISE:
            if text.lower().endswith(" " + noise):
                text = text[: -(len(noise) + 1)].strip(" ,;:")
                changed = True
    return text.strip(" ,;:")


def _split_items(value: str) -> list[str]:
    """Split a multi-value fact into its items ("a, b; c" -> [a, b, c])."""

    return [part.strip() for part in re.split(r"[;,]", value or "") if part.strip()]


def is_question_only(message: str) -> bool:
    """Heuristic: is this turn a question rather than a statement of fact?

    Lab bonus "avoid storing a wrong fact when the user asks instead of tells".
    A question cannot be trusted as new information, so extraction skips it.
    Limit: a fact wrapped inside a rhetorical question would also be skipped.
    """

    text = (message or "").strip()
    if not text:
        return True
    return "?" in text


# --------------------------------------------------------------------------
# Heuristic entity extraction (Vietnamese)
#
# Design rules, driven by the two datasets in `data/`:
#   1. A *correction* must win over the fact it replaces ("giờ ở Huế chứ không
#      còn ở Đà Nẵng"): among all candidates we keep the LAST positive one,
#      ordered by its position in the text.
#   2. *Noise* must never become a fact ("Hà Nội chỉ là nơi vừa bay ra họp",
#      "đùa ... chuyển sang product manager"): candidates inside a negated or
#      joking clause are dropped before rule 1 is applied.
#   3. A *question* is not information ("Bạn có biết DũngCT không?").
# --------------------------------------------------------------------------

# Known Vietnamese place names, longest first so "TP. Hồ Chí Minh" wins.
_CITIES: tuple[str, ...] = (
    "TP. Hồ Chí Minh",
    "TP.Hồ Chí Minh",
    "Hồ Chí Minh",
    "Đà Nẵng",
    "Hà Nội",
    "Nha Trang",
    "Hải Phòng",
    "Cần Thơ",
    "Đà Lạt",
    "Quy Nhơn",
    "Vũng Tàu",
    "Buôn Ma Thuột",
    "Huế",
    "Sài Gòn",
)
# NOTE: must be wrapped in a group, otherwise the `|` binds to the whole
# concatenation and a bare city would match everywhere.
_CITY_ALT = "(?:" + "|".join(re.escape(city) for city in _CITIES) + ")"

# Job titles, used to recognize a profession mention.
_ROLES = (
    "(?:engineer|developer|programmer|architect|analyst|scientist|designer"
    "|researcher|consultant|manager|freelancer|intern|lead|director)"
)

# Markers that make a place mention a *current home*, not a passing mention.
_LOCATION_PATTERNS: tuple[str, ...] = (
    r"chuyển\s+từ\s+[^,;.]+?\s+sang\s+(CITY)",
    r"nơi\s+ở\s+hiện\s+tại\s+(?:là|:\s*)?(CITY)",
    r"(?:hiện|đang)\s+(?:làm\s+việc\s+)?ở\s+(CITY)",
    r"làm\s+việc\s+ở\s+(CITY)",
    r"(?:mình\s+)?vẫn\s+ở\s+(CITY)",
    r"mình\s+ở\s+(CITY)",
    r"ở\s+(CITY)",
)
_LOCATION_NOISE = (
    r"không\s+còn\s+ở\s+" + _CITY_ALT,
    r"không\s+phải\s+(?:nơi\s+ở\s+)?(?:hiện\s+tại\s+)?(?:ở\s+)?" + _CITY_ALT,
    r"chỉ\s+là\s+nơi[^.;!?]{0,40}?" + _CITY_ALT,
    r"bay\s+ra\s+họp|đi\s+họp|họp\s+ở",
    r"(?:trước\s+đó|vừa)\s+có\s+nhắc\s+" + _CITY_ALT,
    r"nhắc\s+lại\s+" + _CITY_ALT,
    r"đừng\s+(?:lấy|dùng)[^.;!?]{0,60}",
)

_PROFESSION_PATTERN = (
    r"(?:chuyển\s+sang|làm|giờ\s+là|hiện\s+là|vẫn\s+là|nghề\s+nghiệp[^.;!?]{0,20}?là)"
    r"\s+((?:[A-Za-z][\w+#.-]*\s+){0,2}" + _ROLES + r")\b"
)
_PROFESSION_NOISE = (r"đùa", r"chỉ\s+là\s+câu", r"nếu\s+[^.;!?]{0,40}nhắc")

# A handle is usually written with several capitals ("DũngCT"), while a pet
# name ("Bơ") or an ordinary word ("phành gia") is not. The second pattern
# only accepts an all-caps-ish token, so "nuôi bé corgi tên Bơ" and "cái tên
# phành gia" are both rejected.
_NAME_STEM = r"([A-Za-zÀ-ỹ][\wÀ-ỹ]*(?:[ -][A-ZÀ-Ỹ][\wÀ-Ỹ]*){0,3})"
_NAME_PATTERNS: tuple[str, ...] = (
    # `\s+` is required before the name, otherwise the stem would try to match
    # the space after "là".
    r"(?:mình\s+tên\s+là|tên\s+mình\s+là|tên\s+là)\s+" + _NAME_STEM,
    # The token right after "tên " must start with an ASCII capital and contain two
    # adjacent ASCII capitals somewhere inside it: "DũngCT" passes, while
    # "Bơ" (pet name), "phành gia" and "là"/"mình" (grammar) do not.
    # `(?-i:...)` keeps the test case sensitive despite IGNORECASE.
    r"tên\s+(?=(?-i:[A-Z])(?=[A-Za-zÀ-ỹ]{0,15}(?-i:[A-Z]{2})))" + _NAME_STEM,
)

_PET_PATTERN = (
    r"nuôi\s+(?:một\s+(?:bé|con|chú|cái)\s+)?((?:[a-zà-ỹ]+\s+){0,2}"
    r"(?:corgi|pug|chihuahua|husky|golden|retriever|mèo|chó|chim|hamster))"
)

# Drink vocabulary: matching a known drink name is far safer than capturing
# everything after the word "cà phê" ("pha cà phê rồi đọc issue mới").
_DRINKS: tuple[str, ...] = (
    "cà phê sữa đá",
    "cà phê sữa nóng",
    "cà phê đen",
    "cà phê nóng",
    "cà phê lon",
    "cà phê túi",
    "cà phê",
    "trà đá",
    "trà sữa",
    "trà sen",
    "bạc xỉu",
    "nước cam",
    "nước ngọt",
    "sữa tươi",
    "sữa chua",
    "americano",
    "latte",
    "cappuccino",
    "mocha",
)
_DRINK_ALT = "(" + "|".join(re.escape(d) for d in _DRINKS) + ")"
# A drink is only a *preference* when the sentence says so; "mình làm việc ở
# quán cà phê" must not overwrite the real favourite drink.
_DRINK_INTENT = r"(?:yêu|ưa|thích|hay|uống|đồ\s+uống)"
_DRINK_PATTERNS: tuple[str, ...] = (
    r"(?:đồ\s+uống)\s*(?:yêu|ưa)?\s*thích\s*(?:là|:)\s*([^.;!?]{2,40}?)(?=[.;!?]|$)",
    _DRINK_ALT,
)
_DRINK_NOISE = _DRINK_INTENT

_FOOD_PATTERNS: tuple[str, ...] = (
    r"món\s+ăn\s+yêu\s*thích\s*(?:là|:)\s*([^.;!?]{2,40}?)(?=[.;!?]|$)",
    r"(mì\s+Quảng|phở\s+[A-Za-zÀ-ỹ]+|bún\s+[A-Za-zÀ-ỹ]+)",
)

# Response-style markers. The stored text must literally contain the phrases
# the benchmark looks for (`ngắn gọn`, `3 bullet`).
_STYLE_MARKERS: tuple[tuple[str, str], ...] = (
    (r"3\s*bullet", "3 bullet"),
    (r"ngắn\s*gọn", "trả lời ngắn gọn"),
    (r"thành\s*bullet|bullet\s+ngắn|dạng\s+bullet", "trả lời dạng bullet"),
    (r"ví\s*dụ\s*(?:thực\s*chiến|thực\s*tế|số\s*liệu)", "có ví dụ thực chiến"),
    (r"trade-?off", "ưu tiên nhấn trade-off"),
)

# Technical interests, a bounded vocabulary so the profile cannot bloat.
_INTEREST_KEYWORDS: tuple[str, ...] = (
    "Python",
    "MLOps",
    "DevOps",
    "LLM",
    "RAG",
    "AI",
    "LangChain",
    "LangGraph",
    "Docker",
    "Kubernetes",
    "FastAPI",
    "React",
    "pytest",
)

# Facts that are replaced (never unioned) so corrections always win.
SINGLE_VALUE_FACTS: frozenset[str] = frozenset(
    {"name", "location", "profession", "favorite_drink", "favorite_food", "pet"}
)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split text into (start, end) sentence spans."""

    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?;:\n]+", text):
        end = match.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _noise_spans(text: str, patterns: tuple[str, ...]) -> list[tuple[int, int]]:
    """Character spans marked as noise / negation for one fact type."""

    return [
        match.span()
        for pattern in patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]


def _sentence_noise_spans(text: str, keywords: tuple[str, ...]) -> list[tuple[int, int]]:
    """Whole-sentence spans that look like jokes or hypotheticals."""

    return [
        (start, end)
        for start, end in _sentence_spans(text)
        if any(re.search(kw, text[start:end], flags=re.IGNORECASE) for kw in keywords)
    ]


def _overlaps(span: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    """Does `span` intersect any of `others`?"""

    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in others)


def _pick_last(
    text: str,
    patterns: tuple[str, ...],
    noise: list[tuple[int, int]],
) -> str | None:
    """Return the value of the last positive, non-noise match in `text`.

    This is what makes a correction beat the fact it replaces: the newest
    mention is the one the user typed last.
    """

    best: str | None = None
    best_start = -1
    for pattern in patterns:
        compiled = pattern.replace("CITY", _CITY_ALT)
        for match in re.finditer(compiled, text, flags=re.IGNORECASE):
            if _overlaps(match.span(), noise):
                continue
            # Patterns must expose the value as group 1; fall back to the whole
            # match so a group-less pattern still behaves.
            value = (match.group(1) if match.groups() else match.group(0)).strip()
            if not value or value.lower() in _QUESTION_WORDS:
                continue
            if match.start() >= best_start:
                best_start = match.start()
                best = value
    return best


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts extracted from the lab datasets:
    - name, location, profession, response_style, favorite_drink,
      favorite_food, pet, interests

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.

    Multi-value facts (`response_style`, `interests`) are stored as fragments
    so preferences accumulate; single-value facts are returned so the caller can
    replace the previous value when the user corrects it.
    """

    text = (message or "").strip()
    if not text or is_question_only(text):
        return {}

    updates: dict[str, str] = {}

    name = _pick_last(text, _NAME_PATTERNS, [])
    if name:
        updates["name"] = name

    profession = _pick_last(
        text,
        (_PROFESSION_PATTERN,),
        _sentence_noise_spans(text, _PROFESSION_NOISE),
    )
    if profession:
        updates["profession"] = _clean_value(profession)

    location = _pick_last(text, _LOCATION_PATTERNS, _noise_spans(text, _LOCATION_NOISE))
    if location:
        updates["location"] = _clean_value(location)

    # A drink only counts when the user expresses a preference for it.
    if re.search(_DRINK_INTENT, text, flags=re.IGNORECASE):
        drink = _pick_last(text, _DRINK_PATTERNS, [])
        if drink:
            updates["favorite_drink"] = _clean_value(drink)

    food = _pick_last(text, _FOOD_PATTERNS, [])
    if food:
        updates["favorite_food"] = _clean_value(food)

    pet = _pick_last(text, (_PET_PATTERN,), [])
    if pet:
        # Keep the animal itself, not its name ("nuôi một bé corgi tên Bơ").
        updates["pet"] = _clean_value(pet).split()[-1]

    styles = [
        canonical
        for pattern, canonical in _STYLE_MARKERS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    if styles:
        updates["response_style"] = "; ".join(styles)

    interests = [
        word
        for word in _INTEREST_KEYWORDS
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text)
    ]
    if interests:
        updates["interests"] = ", ".join(interests)

    return {key: value for key, value in updates.items() if value}


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.

    Current implementation keeps the first meaningful sentence of each message
    and truncates it, so compaction keeps the *shape* of the thread at a much
    lower token cost. It is lossy by design, which is the trade-off the lab
    wants students to observe.
    """

    if not messages:
        return ""

    snippets: list[str] = []
    for message in messages[-max_items:]:
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        # First sentence only, then hard-truncate.
        first = re.split(r"(?<=[.!?])\s+", content, maxsplit=1)[0].strip()
        if len(first) > 140:
            first = first[:137].rstrip() + "..."
        role = "user" if message.get("role") == "user" else "assistant"
        snippets.append(f"{role}: {first}")

    if not snippets:
        return ""
    return " | ".join(snippets)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        # TODO:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        state = self._state_for(thread_id)
        messages = state["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        self._maybe_compact(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        # TODO: return per-thread state with keys like messages, summary, compactions.
        state = self._state_for(thread_id)
        messages = state["messages"]
        assert isinstance(messages, list)
        summary = str(state["summary"])
        return {
            "messages": [dict(message) for message in messages],
            "summary": summary,
            "compactions": int(state["compactions"]),
            "total_messages": int(state["total_messages"]),
            "tokens": self.thread_tokens(thread_id),
        }

    def compaction_count(self, thread_id: str) -> int:
        # TODO: return number of compactions for this thread.
        return int(self._state_for(thread_id)["compactions"])

    def thread_tokens(self, thread_id: str) -> int:
        """Estimated size of the live context (summary + kept messages)."""

        return estimate_tokens(self.render(thread_id))

    def render(self, thread_id: str) -> str:
        """Flatten a thread into the text that would be sent to the model."""

        state = self._state_for(thread_id)
        messages = state["messages"]
        assert isinstance(messages, list)
        parts: list[str] = []
        summary = str(state["summary"])
        if summary:
            parts.append(f"[summary] {summary}")
        parts += [
            f"{message['role']}: {message['content']}" for message in messages
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------- internals
    def _state_for(self, thread_id: str) -> dict[str, object]:
        state = self.state.get(thread_id)
        if state is None:
            state = {
                "messages": [],
                "summary": "",
                "compactions": 0,
                "total_messages": 0,
            }
            self.state[thread_id] = state
        return state

    def _maybe_compact(self, thread_id: str) -> bool:
        """Compress older messages once the thread crosses the threshold."""

        state = self._state_for(thread_id)
        messages = state["messages"]
        assert isinstance(messages, list)
        state["total_messages"] = int(state["total_messages"]) + 1

        keep = max(1, int(self.keep_messages))
        compacted = False

        # Loop, because an already long summary can push us over again.
        while estimate_tokens(self.render(thread_id)) > self.threshold_tokens:
            if len(messages) <= keep:
                break
            older, messages = messages[:-keep], messages[-keep:]
            chunk = summarize_messages(older)
            if chunk:
                previous = str(state["summary"])
                state["summary"] = f"{previous} | {chunk}" if previous else chunk
            state["messages"] = messages
            state["compactions"] = int(state["compactions"]) + 1
            compacted = True

        return compacted
