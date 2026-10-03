from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    BASE_SYSTEM_PROMPT,
    SYSTEM_PROMPT_TOKENS,
    apply_updates_to_facts,
    estimate_tokens,
    extract_profile_updates,
    is_recall_question,
    render_ack,
    render_recall_answer,
)
from model_provider import build_chat_model, has_live_credentials, message_text, usage_totals


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0
    # Scratch facts derived from THIS thread's messages only. They are the baseline's short-term
    # memory (equivalent to the model "reading" its own chat history) and are never persisted.
    facts: dict[str, str] = field(default_factory=dict)


class BaselineAgent:
    """Agent A: within-session memory only.

    - remembers everything said in the *same* thread (the whole history is re-sent every turn)
    - has no `User.md` and no compaction
    - a new thread id starts from zero, so it forgets every long-term fact

    It uses the same fact extractor and answer composer as the advanced agent, so the benchmark
    differences come purely from the memory architecture, not from a weaker "brain".
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.live_error: str | None = None

        # Live mode is optional: stays None (=> deterministic offline mode) without a valid key,
        # without the LangChain packages, or when `force_offline=True`.
        self.langchain_agent = self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return `{"response", "agent_tokens", "prompt_tokens", "mode"}` for one user turn.

        `user_id` is accepted for interface parity with the advanced agent but deliberately unused:
        the baseline has no per-user memory.
        """

        if self.langchain_agent is not None:
            try:
                return self._reply_live(thread_id, message)
            except Exception as exc:  # network, auth, quota, SDK changes... never crash the lab
                self.live_error = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    f"Baseline live mode failed ({self.live_error}); falling back to offline mode.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.langchain_agent = None
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Cumulative agent tokens (user message + reply) for one thread."""

        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Cumulative prompt tokens processed: the full history is re-read on every turn."""

        return self._session(thread_id).prompt_tokens_processed

    def memory_file_size(self, user_id: str) -> int:
        return 0  # no persistent memory file

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    # ---- internals ----------------------------------------------------------------

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    @staticmethod
    def _prompt_tokens(state: SessionState) -> int:
        return SYSTEM_PROMPT_TOKENS + sum(estimate_tokens(m["content"]) for m in state.messages)

    def _record_turn(
        self, state: SessionState, message: str, response: str, prompt_tokens: int, mode: str
    ) -> dict[str, Any]:
        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        state.messages.append({"role": "assistant", "content": response})
        state.token_usage += agent_tokens
        state.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "mode": mode,
        }

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic baseline.

        1. store the user message in this thread
        2. update scratch facts from this thread only
        3. answer recall questions from those scratch facts, otherwise acknowledge
        4. update token counters (prompt = system + the entire thread so far)

        A different `thread_id` gets a fresh SessionState, so nothing carries over.
        """

        state = self._session(thread_id)
        state.messages.append({"role": "user", "content": message})
        apply_updates_to_facts(state.facts, extract_profile_updates(message))

        prompt_tokens = self._prompt_tokens(state)
        if is_recall_question(message):
            response = render_recall_answer(message, state.facts, "cuộc trò chuyện này")
        else:
            response = render_ack({})
        return self._record_turn(state, message, response, prompt_tokens, "offline")

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self._session(thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        result = self.langchain_agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)

        messages = result["messages"]
        response = message_text(messages[-1].content).strip() or "(không có phản hồi)"
        last_human = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"), default=0)
        input_tokens, _ = usage_totals(messages[last_human:])

        # Mirror the exchange locally so (a) token estimates work and (b) a mid-run fallback to
        # offline mode keeps the thread's history.
        state.messages.append({"role": "user", "content": message})
        apply_updates_to_facts(state.facts, extract_profile_updates(message))
        prompt_tokens = input_tokens or self._prompt_tokens(state)
        return self._record_turn(state, message, response, prompt_tokens, "live")

    def _maybe_build_langchain_agent(self):
        """Build a LangChain `create_agent` with an `InMemorySaver` (short-term memory per thread).

        Returns None — and the agent runs offline — when forced offline, when no valid credentials
        exist for the configured provider, or when the LangChain/provider packages are missing.
        """

        if self.force_offline or not has_live_credentials(self.config.model):
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            return create_agent(
                model=build_chat_model(self.config.model),
                tools=[],
                system_prompt=BASE_SYSTEM_PROMPT,
                checkpointer=InMemorySaver(),
            )
        except Exception as exc:  # ImportError, bad config, SDK version drift...
            self.live_error = f"{type(exc).__name__}: {exc}"
            return None
