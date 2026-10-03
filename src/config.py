from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import (
    API_KEY_ENV_BY_PROVIDER,
    BASE_URL_ENV_BY_PROVIDER,
    DEFAULT_MODEL_BY_PROVIDER,
    SUPPORTED_PROVIDERS,
    ProviderConfig,
    normalize_provider,
)


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig

    @property
    def profiles_dir(self) -> Path:
        """Where per-user `User.md` files live."""

        return self.state_dir / "profiles"

    @property
    def standard_dataset(self) -> Path:
        """Standard benchmark: many short conversations + recall questions."""

        return self.data_dir / "conversations.json"

    @property
    def stress_dataset(self) -> Path:
        """Long-context stress benchmark: one very long conversation."""

        return self.data_dir / "advanced_long_context.json"


def _load_dotenv_file(root: Path) -> None:
    """Load `root/.env` when python-dotenv is available.

    Best effort: the lab must run without python-dotenv installed, so a missing
    package or a missing file is simply ignored.
    """

    env_file = root / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_file, override=False)


def _env_int(name: str, default: int) -> int:
    """Read a positive integer env var, falling back to `default`."""

    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to `default`."""

    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _build_provider_config(
    provider: str,
    model_name: str | None,
    temperature: float,
) -> ProviderConfig:
    """Resolve api key / base url for a provider from the environment."""

    provider = normalize_provider(provider)
    return ProviderConfig(
        provider=provider,
        model_name=model_name or DEFAULT_MODEL_BY_PROVIDER[provider],
        temperature=temperature,
        api_key=os.getenv(API_KEY_ENV_BY_PROVIDER[provider]) or None,
        base_url=os.getenv(BASE_URL_ENV_BY_PROVIDER[provider]) or None,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # 2. Optional `.env` at the repo root (already git-ignored).
    _load_dotenv_file(root)

    # 1. Supported providers: openai, custom, gemini, anthropic, ollama, openrouter.
    provider = os.getenv("LLM_PROVIDER", "").strip() or SUPPORTED_PROVIDERS[0]
    temperature = _env_float("LLM_TEMPERATURE", 0.0)

    model = _build_provider_config(provider, os.getenv("LLM_MODEL"), temperature)

    # The judge model may use a different provider / model than the main agent.
    judge_provider = os.getenv("JUDGE_PROVIDER", "").strip() or provider
    judge_model = _build_provider_config(
        judge_provider,
        os.getenv("JUDGE_MODEL"),
        _env_float("JUDGE_TEMPERATURE", 0.0),
    )

    # 3. Make sure `state/` exists so User.md can be written immediately.
    state_dir = root / os.getenv("STATE_DIR", "state").strip()
    state_dir.mkdir(parents=True, exist_ok=True)

    # 4. Sensible compact-memory defaults: compact only once a thread is long
    #    enough to actually hurt, but keep enough recent turns for coherence.
    compact_threshold_tokens = _env_int("COMPACT_THRESHOLD_TOKENS", 600)
    compact_keep_messages = _env_int("COMPACT_KEEP_MESSAGES", 6)

    return LabConfig(
        base_dir=root,
        data_dir=root / os.getenv("DATA_DIR", "data").strip(),
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )
