from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    BASE_SYSTEM_PROMPT,
    FIELD_ORDER,
    SYSTEM_PROMPT_TOKENS,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    is_recall_question,
    render_ack,
    render_recall_answer,
)
from model_provider import build_chat_model, has_live_credentials, message_text, usage_totals

# LangChain is optional. These names are imported at module level (not inside the builder) because
# `from __future__ import annotations` makes annotations strings that `@tool` resolves from here.
try:  # pragma: no cover - exercised only when langchain is installed
    from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
    from langchain.tools import ToolRuntime, tool
except Exception:  # ImportError or version drift: live mode simply stays unavailable
    ModelRequest = SummarizationMiddleware = dynamic_prompt = ToolRuntime = tool = None  # type: ignore[assignment]

# Writing to User.md through a tool call costs the agent tokens that the offline path would not
# otherwise pay, so we charge a fixed call overhead plus the payload size.
MEMORY_WRITE_OVERHEAD_TOKENS = 8
MAX_TOOL_VALUE_CHARS = 120
_SUMMARY_MARKER = "Here is a summary of the conversation to date"


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent with three memory layers.

    1. within-session (short-term) memory: the recent messages of the thread
    2. persistent memory: `User.md`, written from stable facts and read in every new thread
    3. compact memory: older messages are folded into a bounded summary once the thread is too long

    Prompt = system prompt + User.md + summary + recent messages.
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
        self.live_error: str | None = None
        self._live_compactions: dict[str, int] = {}
        self._live_summary_seen: dict[str, str] = {}

        # Optional live agent; None => deterministic offline mode.
        self.langchain_agent = self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between live and offline mode; live failures degrade to offline instead of raising."""

        if self.langchain_agent is not None:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception as exc:  # network, auth, quota, SDK changes... never crash the lab
                self.live_error = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    f"Advanced live mode failed ({self.live_error}); falling back to offline mode.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.langchain_agent = None
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        live = self._live_compactions.get(thread_id)
        return live if live is not None else self.compact_memory.compaction_count(thread_id)

    # ---- offline path -------------------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path.

        1. extract stable profile facts from the incoming message (confidence-filtered)
        2. persist them into `User.md` (a correction replaces the stale value)
        3. append the message to compact memory (may trigger a compaction)
        4. estimate the prompt context: system + `User.md` + summary + recent messages
        5. answer from persisted memory
        6. append the reply and update token counters
        """

        updates = extract_profile_updates(message)
        changes = self.profile_store.apply_updates(user_id, updates)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        response = self._offline_response(user_id, thread_id, message, changes)
        self.compact_memory.append(thread_id, "assistant", response)

        return self._record_turn(thread_id, message, response, prompt_tokens, changes, "offline")

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: system prompt + User.md + compact summary + recent messages."""

        return (
            SYSTEM_PROMPT_TOKENS
            + estimate_tokens(self.profile_store.read_text(user_id))
            + self.compact_memory.context_tokens(thread_id)
        )

    def _offline_response(
        self, user_id: str, thread_id: str, message: str, changes: dict[str, str] | None = None
    ) -> str:
        """Answer from persisted memory: recall questions read `User.md`, other turns are acknowledged."""

        if is_recall_question(message):
            return render_recall_answer(message, self.profile_store.facts(user_id), "User.md")
        return render_ack(changes or {}, "User.md")

    def _record_turn(
        self,
        thread_id: str,
        message: str,
        response: str,
        prompt_tokens: int,
        changes: dict[str, str],
        mode: str,
    ) -> dict[str, Any]:
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        if changes:
            payload = "; ".join(f"{name}: {value}" for name, value in changes.items())
            agent_tokens += MEMORY_WRITE_OVERHEAD_TOKENS + estimate_tokens(payload)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "mode": mode,
            "memory_changes": changes,
        }

    # ---- live path ----------------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Rule-based extraction stays on as a guardrail, so persistent memory does not depend on the
        # model deciding to call the write tool. It is idempotent, so a later offline retry is safe.
        changes = self.profile_store.apply_updates(user_id, extract_profile_updates(message))

        context = AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)))
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=context,
        )
        messages = result["messages"]
        response = message_text(messages[-1].content).strip() or "(không có phản hồi)"
        last_human = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"), default=0)
        input_tokens, _ = usage_totals(messages[last_human:])
        self._track_live_compaction(thread_id, messages)

        # Only touch local state after the live call succeeded, so a failure falls back cleanly.
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = input_tokens or self._estimate_prompt_context_tokens(user_id, thread_id)
        self.compact_memory.append(thread_id, "assistant", response)
        return self._record_turn(thread_id, message, response, prompt_tokens, changes, "live")

    def _track_live_compaction(self, thread_id: str, messages: list[Any]) -> None:
        """Count real SummarizationMiddleware compactions: it replaces old turns with a summary message."""

        count = self._live_compactions.setdefault(thread_id, 0)
        if not messages:
            return
        first = message_text(getattr(messages[0], "content", ""))
        if first.startswith(_SUMMARY_MARKER) and first != self._live_summary_seen.get(thread_id):
            self._live_summary_seen[thread_id] = first
            self._live_compactions[thread_id] = count + 1

    def _maybe_build_langchain_agent(self):
        """Wire a live agent: provider model + InMemorySaver + User.md tools + dynamic prompt +
        summarization middleware.

        Returns None — and the agent runs offline — when forced offline, when no valid credentials
        exist for the configured provider, or when LangChain / the provider SDK is unavailable.
        """

        if self.force_offline or not has_live_credentials(self.config.model):
            return None
        if tool is None or SummarizationMiddleware is None:
            self.live_error = "langchain is not installed (pip install langchain langgraph)"
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            model = build_chat_model(self.config.model)
            store = self.profile_store

            @tool
            def read_user_profile(runtime: ToolRuntime[AgentContext]) -> str:
                """Read the persistent User.md profile of the current user."""

                return store.read_text(runtime.context.user_id)

            @tool
            def update_user_profile(field: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
                """Save or correct ONE stable fact about the user in User.md.

                `field` must be one of: name, location, profession, favorite_drink, favorite_food,
                pet, interests, hobbies, style. A new value replaces the old one for single-valued
                fields. Only call this for durable facts the user stated about themselves, never
                for questions, jokes, or one-off details.
                """

                if field not in FIELD_ORDER:
                    return f"Unknown field {field!r}. Allowed: {', '.join(FIELD_ORDER)}"
                value = " ".join(value.split())
                if not value or len(value) > MAX_TOOL_VALUE_CHARS:
                    return f"Value must be 1-{MAX_TOOL_VALUE_CHARS} characters."
                changed = store.apply_updates(runtime.context.user_id, {field: value})
                return f"Saved {field} = {value}" if changed else f"{field} already up to date."

            @dynamic_prompt
            def profile_prompt(request: ModelRequest) -> str:
                user_id = request.runtime.context.user_id
                return f"{BASE_SYSTEM_PROMPT}\n\n## User.md (persistent memory)\n{store.read_text(user_id)}"

            threshold = self.config.compact_threshold_tokens
            keep = self.config.compact_keep_messages
            try:
                summarizer = SummarizationMiddleware(
                    model=model, trigger=("tokens", threshold), keep=("messages", keep)
                )
            except TypeError:  # older langchain 1.0.x signature
                summarizer = SummarizationMiddleware(
                    model=model, max_tokens_before_summary=threshold, messages_to_keep=keep
                )

            return create_agent(
                model=model,
                tools=[read_user_profile, update_user_profile],
                middleware=[profile_prompt, summarizer],
                context_schema=AgentContext,
                checkpointer=InMemorySaver(),
            )
        except Exception as exc:  # ImportError, bad config, SDK version drift...
            self.live_error = f"{type(exc).__name__}: {exc}"
            return None
