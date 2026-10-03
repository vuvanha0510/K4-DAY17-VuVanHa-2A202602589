from __future__ import annotations

from dataclasses import dataclass

# Every provider the lab is allowed to run on.
SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
)

# Default chat model per provider, used when the env does not specify one.
DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

# Env var holding the API key / base URL, per provider.
API_KEY_ENV_BY_PROVIDER: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

BASE_URL_ENV_BY_PROVIDER: dict[str, str] = {
    "openai": "OPENAI_BASE_URL",
    "custom": "CUSTOM_BASE_URL",
    "gemini": "GEMINI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}

# Common misspellings / alternate spellings a student may type.
_PROVIDER_ALIASES: dict[str, str] = {
    "anthorpic": "anthropic",
    "claude": "anthropic",
    "google": "gemini",
    "google_genai": "gemini",
    "google-genai": "gemini",
    "googlegenai": "gemini",
    "gemini_ai": "gemini",
    "chatgpt": "openai",
    "gpt": "openai",
    "oai": "openai",
    "open-ai": "openai",
    "open_router": "openrouter",
    "open-router": "openrouter",
    "router": "openrouter",
    "custom_openai": "custom",
    "openai_compatible": "custom",
    "openai-compatible": "custom",
    "local": "ollama",
    "olama": "ollama",
}


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    @property
    def is_supported(self) -> bool:
        return self.provider in SUPPORTED_PROVIDERS


def normalize_provider(value: str) -> str:
    """Normalize a provider name.

    Handles casing, whitespace, separators and common misspellings such as
    `anthorpic` -> `anthropic`.

    Raises:
        ValueError: when the provider is not one of `SUPPORTED_PROVIDERS`.
    """

    if value is None or not str(value).strip():
        raise ValueError(
            "Provider must not be empty. Supported providers: "
            + ", ".join(SUPPORTED_PROVIDERS)
        )

    raw = str(value).strip().lower().replace(" ", "")

    candidate = _PROVIDER_ALIASES.get(raw, raw)
    if candidate in SUPPORTED_PROVIDERS:
        return candidate

    # Last chance: match on a prefix, e.g. `openai-gpt4` -> `openai`.
    for supported in SUPPORTED_PROVIDERS:
        if candidate.startswith(supported):
            return supported

    raise ValueError(
        f"Unsupported provider: {value!r}. Supported providers: "
        + ", ".join(SUPPORTED_PROVIDERS)
    )


def default_model_for(provider: str) -> str:
    """Return the default chat model name for a provider."""

    return DEFAULT_MODEL_BY_PROVIDER[normalize_provider(provider)]


def _missing_package(provider: str, package: str) -> ImportError:
    return ImportError(
        f"Provider {provider!r} needs the {package!r} package. "
        f"Install it with: pip install {package}"
    )


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Provider SDKs are imported lazily so importing this module (and running the
    whole lab offline) never requires LangChain to be installed.

    Mapping:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise _missing_package(provider, "langchain-openai") from exc
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "custom":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise _missing_package(provider, "langchain-openai") from exc
        if not config.base_url:
            raise ValueError(
                "Provider 'custom' requires a base URL "
                "(set CUSTOM_BASE_URL or pass ProviderConfig.base_url)."
            )
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise _missing_package(provider, "langchain-google-genai") from exc
        kwargs: dict[str, object] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depends on env
            raise _missing_package(provider, "langchain-anthropic") from exc
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - depends on env
            raise _missing_package(provider, "langchain-ollama") from exc
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    # provider == "openrouter"
    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError as exc:  # pragma: no cover - depends on env
        raise _missing_package(provider, "langchain-openrouter") from exc
    kwargs = {
        "model": config.model_name,
        "temperature": config.temperature,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenRouter(**kwargs)
