from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_PROVIDER_ALIASES = {
    "openai": "openai",
    "oai": "openai",
    "gpt": "openai",
    "chatgpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "googlegenai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "anthropc": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# `.env.example`-style values people forget to replace; they must not count as real keys.
_PLACEHOLDER_MARKERS = ("...", "your", "xxx", "<", "changeme", "replace", "todo", "placeholder")


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`; raise ValueError for unknown names."""

    key = (value or "").strip().lower()
    if key in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[key]
    raise ValueError(
        f"Unsupported provider {value!r}. Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def is_real_secret(value: str | None) -> bool:
    """False for empty values and obvious placeholders such as `...` or `your_api_key_here`."""

    if value is None:
        return False
    text = value.strip().lower()
    if not text:
        return False
    return not any(marker in text for marker in _PLACEHOLDER_MARKERS)


def _ollama_reachable(base_url: str | None, timeout: float = 0.3) -> bool:
    parsed = urlparse(base_url or DEFAULT_OLLAMA_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 11434)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def has_live_credentials(config: ProviderConfig) -> bool:
    """Cheap, network-light check: does this config look usable for a live LLM call?

    - openai / gemini / anthropic / openrouter need a real (non-placeholder) API key
    - custom needs a base URL (the key is optional for many local gateways)
    - ollama needs no key, but a server must answer on its port
    """

    provider = normalize_provider(config.provider)
    if provider == "ollama":
        return _ollama_reachable(config.base_url)
    if provider == "custom":
        return bool((config.base_url or "").strip())
    return is_real_secret(config.api_key)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Provider SDKs are imported lazily so the offline lab runs without them installed.
    Raises ImportError (missing SDK) or ValueError (bad config); callers that must never
    crash catch both and fall back to offline mode.
    """

    provider = normalize_provider(config.provider)
    common: dict[str, Any] = {"model": config.model_name, "temperature": config.temperature}

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs = dict(common)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if provider == "custom":
            if not config.base_url:
                raise ValueError("provider 'custom' requires a base_url (CUSTOM_BASE_URL)")
            kwargs["base_url"] = config.base_url
            kwargs.setdefault("api_key", "not-needed")
        elif config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(api_key=config.api_key, **common)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(api_key=config.api_key, **common)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(base_url=config.base_url or DEFAULT_OLLAMA_URL, **common)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(api_key=config.api_key, **common)

    raise ValueError(f"Unsupported provider {config.provider!r}")  # pragma: no cover


def usage_totals(messages: list[Any]) -> tuple[int, int]:
    """Sum provider-reported (input_tokens, output_tokens) over AI messages; (0, 0) if unreported."""

    input_tokens = output_tokens = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
    return input_tokens, output_tokens


def message_text(content: Any) -> str:
    """Flatten LangChain message content (str, or a list of content blocks) into plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)
