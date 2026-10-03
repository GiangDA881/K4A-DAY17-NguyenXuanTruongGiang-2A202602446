from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import DEFAULT_OLLAMA_URL, ProviderConfig, is_real_secret, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - paths: repo root, dataset directory, state directory (holds `profiles/<user>/User.md`)
    - compact memory: token threshold that triggers compaction + number of recent messages kept
    - providers: the main chat model and the judge model
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

# Order used to pick a provider automatically when LLM_PROVIDER is not set.
_AUTO_DETECT_ORDER = ("openai", "anthropic", "gemini", "openrouter")

DEFAULT_COMPACT_THRESHOLD_TOKENS = 1000
DEFAULT_COMPACT_KEEP_MESSAGES = 4


def _load_dotenv(path: Path) -> None:
    """Load `path` into os.environ without overriding variables that are already set."""

    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    # python-dotenv is optional: a tiny KEY=VALUE parser keeps offline mode dependency-free.
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _env_number(name: str, default: float, cast=float):
    raw = _env(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except ValueError:
        return default


def _api_key_for(provider: str, prefix: str = "") -> str | None:
    """Look up the API key for `provider`; `prefix` (e.g. `JUDGE_`) allows per-role overrides."""

    if prefix:
        override = _env(f"{prefix}API_KEY")
        if override:
            return override
    if provider == "openai":
        return _env("OPENAI_API_KEY")
    if provider == "gemini":
        return _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if provider == "anthropic":
        return _env("ANTHROPIC_API_KEY")
    if provider == "openrouter":
        return _env("OPENROUTER_API_KEY")
    if provider == "custom":
        return _env("CUSTOM_API_KEY")
    return None  # ollama needs no key


def _base_url_for(provider: str, prefix: str = "") -> str | None:
    if prefix:
        override = _env(f"{prefix}BASE_URL")
        if override:
            return override
    if provider == "custom":
        return _env("CUSTOM_BASE_URL")
    if provider == "ollama":
        return _env("OLLAMA_BASE_URL", default=DEFAULT_OLLAMA_URL)
    if provider == "openai":
        return _env("OPENAI_BASE_URL")
    return None


def _detect_provider() -> str:
    """Honour LLM_PROVIDER; otherwise pick the first provider that has a real API key.

    With no keys at all we return `openai` as a harmless default: the agents will see that
    no credentials exist and stay in deterministic offline mode.
    """

    explicit = _env("LLM_PROVIDER")
    if explicit:
        return normalize_provider(explicit)
    for provider in _AUTO_DETECT_ORDER:
        if is_real_secret(_api_key_for(provider)):
            return provider
    return "openai"


def _build_provider(provider: str, model_name: str | None, temperature: float, prefix: str = "") -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model_name=model_name or DEFAULT_MODELS[provider],
        temperature=temperature,
        api_key=_api_key_for(provider, prefix),
        base_url=_base_url_for(provider, prefix),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables (and `.env`) and return a complete LabConfig.

    Never requires an API key: without one the provider config is still returned and the agents
    simply run offline. Supported variables:

        LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE
        OPENAI_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY), ANTHROPIC_API_KEY, OPENROUTER_API_KEY
        CUSTOM_BASE_URL, CUSTOM_API_KEY, OLLAMA_BASE_URL
        JUDGE_PROVIDER, JUDGE_MODEL            (default: same as the main model)
        COMPACT_THRESHOLD_TOKENS, COMPACT_KEEP_MESSAGES
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root / ".env")

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    temperature = _env_number("LLM_TEMPERATURE", 0.0)
    provider = _detect_provider()
    model = _build_provider(provider, _env("LLM_MODEL"), temperature)

    judge_name = _env("JUDGE_PROVIDER")
    judge_provider = normalize_provider(judge_name) if judge_name else provider
    judge_model_name = _env("JUDGE_MODEL") or (_env("LLM_MODEL") if judge_provider == provider else None)
    judge_model = _build_provider(judge_provider, judge_model_name, 0.0, prefix="JUDGE_")

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=max(1, int(_env_number("COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS, int))),
        compact_keep_messages=max(1, int(_env_number("COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES, int))),
        model=model,
        judge_model=judge_model,
    )
