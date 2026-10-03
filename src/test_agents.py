from __future__ import annotations

import os
from pathlib import Path

import pytest

import agent_advanced
import agent_baseline
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import (
    COLUMNS,
    format_rows,
    heuristic_quality,
    load_conversations,
    recall_points,
    run_agent_benchmark,
)
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import ProviderConfig, has_live_credentials, normalize_provider

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PROVIDER_ENV_VARS = [
    "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
    "CUSTOM_BASE_URL", "CUSTOM_API_KEY", "OLLAMA_BASE_URL", "JUDGE_PROVIDER", "JUDGE_MODEL",
    "COMPACT_THRESHOLD_TOKENS", "COMPACT_KEEP_MESSAGES",
]


def make_config(tmp_path: Path, threshold: int = 1000, keep: int = 4) -> LabConfig:
    """Isolated config for tests: state lives in tmp_path, never in the repo's `state/`."""

    provider = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=DATA_DIR,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
        model=provider,
        judge_model=provider,
    )


@pytest.fixture
def clean_env():
    """No provider env vars, restored afterwards (load_config may load a `.env` into os.environ)."""

    saved = dict(os.environ)
    for name in PROVIDER_ENV_VARS:
        os.environ.pop(name, None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def feed(agent, user_id: str, thread_id: str, turns: list[str]) -> list[dict]:
    return [agent.reply(user_id, thread_id, turn) for turn in turns]


# ---------------------------------------------------------------------------------------
# User.md (persistent memory)
# ---------------------------------------------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # A user with no file yet gets a default markdown profile and size 0 on disk.
    assert store.read_text("dungct").startswith("# User Profile")
    assert store.file_size("dungct") == 0

    path = store.write_text("dungct", "# User Profile\n\n## Facts\n- name: DũngCT\n- location: Đà Nẵng\n")
    assert path.name == "User.md" and path.is_file()
    assert "DũngCT" in store.read_text("dungct")
    assert store.file_size("dungct") == path.stat().st_size > 0

    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    assert store.facts("dungct")["location"] == "Huế"
    assert "Đà Nẵng" not in store.read_text("dungct")
    assert store.edit_text("dungct", "không có đoạn này", "x") is False  # miss leaves the file alone


def test_user_markdown_upsert_replaces_stale_fact(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    store.write_text("u", "# User Profile\n\n## Facts\n\nGhi chú viết tay: giữ nguyên.\n")

    assert store.upsert_fact("u", "location", "Đà Nẵng") is True
    assert store.upsert_fact("u", "location", "Đà Nẵng") is False  # same value: no change, no rewrite
    assert store.upsert_fact("u", "location", "Huế") is True  # correction replaces, never duplicates

    text = store.read_text("u")
    assert text.count("location:") == 1 and "Đà Nẵng" not in text
    assert "Ghi chú viết tay: giữ nguyên." in text  # hand-written content survives machine edits


def test_user_markdown_less_specific_value_does_not_overwrite(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    store.upsert_fact("u", "profession", "MLOps engineer")
    assert store.upsert_fact("u", "profession", "MLOps") is False
    assert store.facts("u")["profession"] == "MLOps engineer"


def test_user_markdown_path_is_sanitized(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    for hostile in ("../../etc/passwd", "a/b", "..", ""):
        path = store.path_for(hostile).resolve()
        assert (tmp_path / "profiles").resolve() in path.parents
    assert store.path_for("a b") != store.path_for("a_b")  # sanitising must not merge two users


def test_list_fields_are_bounded(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    for i in range(30):
        store.upsert_fact("u", "interests", f"chủ đề {i}")
    interests = store.facts("u")["interests"].split(", ")
    assert len(interests) <= 8 and interests[-1] == "chủ đề 29"  # newest kept, oldest decayed
    assert store.file_size("u") < 500


# ---------------------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Chào bạn, mình tên là DũngCT.", {"name": "DũngCT"}),
        ("Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
         {"location": "Đà Nẵng", "profession": "backend engineer"}),
        # correction: new location wins, "không còn ở Đà Nẵng" is not read as a statement
        ("À, mình đính chính một chút: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.",
         {"location": "Huế"}),
        ("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
         {"profession": "MLOps engineer"}),
        ("Bạn nhớ là nơi ở đã cập nhật từ Huế sang Đà Nẵng, còn các preference khác vẫn giữ.",
         {"location": "Đà Nẵng"}),
        # the old side of a correction ("Lúc đầu ... Huế") must be ignored
        ("Lúc đầu mình nói hiện ở Huế, nhưng thực ra từ tuần này mình đang làm việc ở Đà Nẵng vài tháng.",
         {"location": "Đà Nẵng"}),
        # noise: joke, and a place that is only a business trip
        ("Có lúc mình đùa rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa.", {}),
        ("Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại.", {}),
        # questions are requests, not facts
        ("Tên mình là gì và mình thích kiểu trả lời như thế nào?", {}),
        ("Hiện tại mình làm nghề gì và mình còn ở Huế không?", {}),
        ("Mình thích Python, AI ứng dụng và cà phê sữa đá.",
         {"favorite_drink": "cà phê sữa đá", "interests": "Python, AI ứng dụng"}),
        ("Mình nuôi một bé corgi tên Bơ.", {"pet": "corgi tên Bơ"}),
        ("Món ăn yêu thích là mì Quảng.", {"favorite_food": "mì Quảng"}),
        # "giải thích" (explain) must not be misread as "thích" (like)
        ("Mỗi khi giải thích benchmark, bạn thử nêu thêm một ví dụ số liệu minh họa.", {"style": "có ví dụ"}),
    ],
)
def test_extract_profile_updates(message: str, expected: dict[str, str]) -> None:
    assert extract_profile_updates(message) == expected


def test_extract_style_is_canonical() -> None:
    updates = extract_profile_updates("Mình muốn bạn trả lời ngắn gọn thành 3 bullet, có ví dụ thực chiến.")
    style = updates["style"]
    assert "ngắn gọn" in style and "3 bullet" in style and "ví dụ thực chiến" in style


def test_confidence_threshold_blocks_weak_signals() -> None:
    habit = "Mình vẫn uống cà phê sữa đá như cũ."
    assert extract_profile_updates(habit) == {}  # a habit is not a stated preference
    assert extract_profile_updates(habit, min_confidence=0.6) == {"favorite_drink": "cà phê sữa đá"}


# ---------------------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------------------


def test_compact_trigger() -> None:
    manager = CompactMemoryManager(threshold_tokens=60, keep_messages=2)
    for index in range(8):
        role = "user" if index % 2 == 0 else "assistant"
        manager.append("t", role, f"Tin nhắn số {index}. " + "nội dung khá dài " * 6)

    context = manager.context("t")
    assert manager.compaction_count("t") >= 1
    assert context["summary"]
    assert "số 0" in context["summary"]  # oldest content was folded into the summary...
    assert all("số 0" not in m["content"] for m in context["messages"])  # ...and left the raw messages
    assert context["messages"][-1]["content"].startswith("Tin nhắn số 7")  # newest stays verbatim


def test_compact_does_not_trigger_below_threshold() -> None:
    manager = CompactMemoryManager(threshold_tokens=500, keep_messages=2)
    for index in range(6):
        manager.append("t", "user", f"ngắn {index}")
    assert manager.compaction_count("t") == 0
    assert len(manager.context("t")["messages"]) == 6


def test_compact_summary_stays_bounded_and_never_loops() -> None:
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2, max_summary_tokens=80)
    for index in range(60):
        manager.append("t", "user", f"Thông tin số {index}: " + "chi tiết " * 12)
    assert estimate_tokens(str(manager.context("t")["summary"])) <= 80

    huge = CompactMemoryManager(threshold_tokens=10, keep_messages=2)
    huge.append("h", "user", "x " * 500)  # a single oversized message cannot be compacted away
    assert huge.compaction_count("h") == 0


def test_advanced_agent_triggers_compaction(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path, threshold=120, keep=2), force_offline=True)
    feed(agent, "u", "long", [f"Tin số {i}. " + "mình kể thêm một đoạn khá dài. " * 4 for i in range(8)])
    assert agent.compaction_count("long") >= 1
    assert BaselineAgent(make_config(tmp_path), force_offline=True).compaction_count("long") == 0


# ---------------------------------------------------------------------------------------
# Cross-session recall (the core behavioural difference)
# ---------------------------------------------------------------------------------------

INTRO = [
    "Chào bạn, mình tên là DũngCT.",
    "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
    "Đồ uống yêu thích là cà phê sữa đá.",
    "Mình muốn bạn trả lời ngắn gọn và có ví dụ thực tế.",
]


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    feed(baseline, "dungct", "session-1", INTRO)
    feed(advanced, "dungct", "session-1", INTRO)

    question = "Mình tên gì và đồ uống yêu thích là gì?"
    new_thread = "session-2"
    advanced_answer = advanced.reply("dungct", new_thread, question)["response"]
    baseline_answer = baseline.reply("dungct", new_thread, question)["response"]

    assert recall_points(advanced_answer, ["DũngCT", "cà phê sữa đá"]) == 1.0
    assert recall_points(baseline_answer, ["DũngCT", "cà phê sữa đá"]) == 0.0  # new thread: baseline forgot

    # ...but the baseline DOES have short-term memory inside the same thread.
    same_thread = baseline.reply("dungct", "session-1", question)["response"]
    assert recall_points(same_thread, ["DũngCT", "cà phê sữa đá"]) == 1.0


def test_persistent_memory_survives_a_new_agent_instance(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    feed(AdvancedAgent(config, force_offline=True), "dungct", "s1", INTRO)

    restarted = AdvancedAgent(config, force_offline=True)  # fresh process: only User.md remains
    answer = restarted.reply("dungct", "s2", "Tên mình là gì?")["response"]
    assert "DũngCT" in answer


def test_memory_is_isolated_per_user(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    feed(agent, "alice", "a1", ["Mình tên là Alice."])
    answer = agent.reply("bob", "b1", "Tên mình là gì?")["response"]
    assert "Alice" not in answer


def test_correction_updates_instead_of_keeping_both(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    feed(agent, "u", "s1", ["Mình ở Đà Nẵng.", "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa."])
    answer = agent.reply("u", "s2", "Hiện tại mình đang ở đâu?")["response"]
    assert "Huế" in answer and "Đà Nẵng" not in answer
    assert "Đà Nẵng" not in agent.profile_store.read_text("u")


def test_noise_and_questions_are_not_persisted(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    feed(
        agent,
        "u",
        "s1",
        [
            "Mình đang làm MLOps engineer và đang ở Đà Nẵng.",
            "Có lúc mình đùa là hay chuyển sang product manager, nhưng đó chỉ là câu đùa.",
            "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại.",
            "Mình ở Hà Nội không nhỉ?",
        ],
    )
    facts = agent.profile_store.facts("u")
    assert facts["profession"] == "MLOps engineer" and facts["location"] == "Đà Nẵng"
    assert "product manager" not in agent.profile_store.read_text("u")
    assert "Hà Nội" not in agent.profile_store.read_text("u")


def test_facts_survive_compaction(tmp_path: Path) -> None:
    """Compaction may drop raw text, but stable facts live in User.md so recall must not degrade."""

    config = make_config(tmp_path, threshold=150, keep=2)
    agent = AdvancedAgent(config, force_offline=True)
    filler = [f"Hôm nay mình đọc tin số {i} khá dài. " + "bối cảnh bổ sung " * 10 for i in range(8)]
    feed(agent, "u", "long", INTRO[:2] + filler)
    assert agent.compaction_count("long") >= 1

    answer = agent.reply("u", "fresh", "Tên mình là gì và mình ở đâu?")["response"]
    assert recall_points(answer, ["DũngCT", "Đà Nẵng"]) == 1.0


# ---------------------------------------------------------------------------------------
# Prompt load: where compaction pays off, and where it does not
# ---------------------------------------------------------------------------------------


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    conversation = load_conversations(DATA_DIR / "advanced_long_context.json")[0]
    config = make_config(tmp_path)  # same defaults as load_config(): threshold 1000, keep 4
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    thread = conversation["id"]
    base_turns = feed(baseline, conversation["user_id"], thread, conversation["turns"])
    adv_turns = feed(advanced, conversation["user_id"], thread, conversation["turns"])

    assert advanced.compaction_count(thread) >= 2  # the stress thread really forces repeated compaction
    assert advanced.prompt_token_usage(thread) < 0.75 * baseline.prompt_token_usage(thread)
    # Baseline cost grows with every turn; the advanced prompt stays bounded.
    assert [t["prompt_tokens"] for t in base_turns] == sorted(t["prompt_tokens"] for t in base_turns)
    assert max(t["prompt_tokens"] for t in adv_turns) < base_turns[-1]["prompt_tokens"]
    # Compaction only saves context, never user-visible conversation tokens.
    assert advanced.token_usage(thread) < 1.2 * baseline.token_usage(thread)


def test_compact_does_not_win_on_short_threads(tmp_path: Path) -> None:
    conversation = load_conversations(DATA_DIR / "conversations.json")[0]
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    feed(baseline, conversation["user_id"], "short", conversation["turns"])
    feed(advanced, conversation["user_id"], "short", conversation["turns"])

    assert advanced.compaction_count("short") == 0  # nothing to compact yet...
    # ...yet User.md is injected into every prompt, so the advanced agent costs more.
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")


# ---------------------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------------------


def test_recall_points_and_quality() -> None:
    assert recall_points("Tên bạn là DũngCT, thích cà phê sữa đá", ["DũngCT", "cà phê sữa đá"]) == 1.0
    assert recall_points("Tên bạn là DũngCT", ["DũngCT", "cà phê sữa đá"]) == 0.5
    assert recall_points("", ["DũngCT"]) == 0.0
    assert recall_points("DũngCT", []) == 0.0
    # the acronym "AI" must not match the Vietnamese words "hai" / "bài" / "ai"
    assert recall_points("hai bài toán, ai biết", ["AI"]) == 0.0
    assert recall_points("Python và AI ứng dụng", ["AI"]) == 1.0
    assert recall_points("mình thích CÀ PHÊ SỮA ĐÁ", ["cà phê sữa đá"]) == 1.0

    assert heuristic_quality("DũngCT", ["DũngCT"]) == 1.0
    assert heuristic_quality("Mình không biết.", ["DũngCT"]) == 0.0
    assert heuristic_quality("DũngCT " + "dài dòng " * 200, ["DũngCT"]) < 1.0


def test_benchmark_separates_baseline_from_advanced(tmp_path: Path) -> None:
    conversations = load_conversations(DATA_DIR / "conversations.json")
    rows = {}
    for name, cls in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        config = make_config(tmp_path / name.lower())
        rows[name] = run_agent_benchmark(name, cls(config, force_offline=True), conversations, config)

    baseline, advanced = rows["Baseline"], rows["Advanced"]
    assert baseline.recall_score == 0.0 and baseline.memory_growth_bytes == 0 and baseline.compactions == 0
    assert advanced.recall_score >= 0.9 and advanced.response_quality >= 0.9
    assert advanced.memory_growth_bytes > 0
    assert advanced.prompt_tokens_processed > baseline.prompt_tokens_processed  # short chats: memory costs

    table = format_rows(list(rows.values()))
    for column in COLUMNS:
        assert column in table


def test_stress_benchmark_shows_compaction_savings(tmp_path: Path) -> None:
    conversations = load_conversations(DATA_DIR / "advanced_long_context.json")
    rows = {}
    for name, cls in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        config = make_config(tmp_path / name.lower())
        rows[name] = run_agent_benchmark(name, cls(config, force_offline=True), conversations, config)

    baseline, advanced = rows["Baseline"], rows["Advanced"]
    assert advanced.recall_score == 1.0 and baseline.recall_score == 0.0
    assert advanced.compactions >= 2
    assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed


def test_llm_judge_is_optional_and_never_crashes(tmp_path: Path, monkeypatch) -> None:
    import benchmark

    config = make_config(tmp_path)
    args = (config, "Mình tên gì?", "Tên bạn là DũngCT", ["DũngCT"])
    assert benchmark.llm_judge_quality(*args) is None  # no credentials -> heuristic fallback

    class StubModel:
        def __init__(self, content: str | Exception) -> None:
            self.content = content

        def invoke(self, _prompt):
            if isinstance(self.content, Exception):
                raise self.content
            return type("Reply", (), {"content": self.content})()

    monkeypatch.setattr(benchmark, "has_live_credentials", lambda _config: True)
    monkeypatch.setattr(benchmark, "build_chat_model", lambda _config: StubModel("Điểm: 0.8"))
    assert benchmark.llm_judge_quality(*args) == 0.8
    monkeypatch.setattr(benchmark, "build_chat_model", lambda _config: StubModel("7"))
    assert benchmark.llm_judge_quality(*args) == 1.0  # clamped into [0, 1]
    monkeypatch.setattr(benchmark, "build_chat_model", lambda _config: StubModel(TimeoutError("down")))
    assert benchmark.llm_judge_quality(*args) is None

    # run_agent_benchmark prefers the judge score when asked to, otherwise the heuristic.
    monkeypatch.setattr(benchmark, "build_chat_model", lambda _config: StubModel("0.5"))
    conversations = load_conversations(DATA_DIR / "conversations.json")[:1]
    advanced = AdvancedAgent(config, force_offline=True)
    judged = run_agent_benchmark("Advanced", advanced, conversations, config, use_judge=True)
    assert judged.response_quality == 0.5 and judged.recall_score == 1.0


def test_benchmark_is_deterministic(tmp_path: Path) -> None:
    conversations = load_conversations(DATA_DIR / "advanced_long_context.json")
    results = []
    for run in ("a", "b"):
        config = make_config(tmp_path / run)
        results.append(run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config))
    assert results[0] == results[1]


# ---------------------------------------------------------------------------------------
# Providers, config, and the live/offline switch (must never crash)
# ---------------------------------------------------------------------------------------


def test_normalize_provider_aliases() -> None:
    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider(" OpenAI ") == "openai"
    assert normalize_provider("google") == "gemini"
    assert normalize_provider("open-router") == "openrouter"
    assert normalize_provider("openai-compatible") == "custom"
    for name in ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter"):
        assert normalize_provider(name) == name
    with pytest.raises(ValueError):
        normalize_provider("skynet")


def test_credentials_check_ignores_placeholders() -> None:
    assert not has_live_credentials(ProviderConfig("openai", "m", 0.0, api_key=None))
    assert not has_live_credentials(ProviderConfig("openai", "m", 0.0, api_key="..."))
    assert not has_live_credentials(ProviderConfig("anthropic", "m", 0.0, api_key="your_api_key_here"))
    assert has_live_credentials(ProviderConfig("openai", "m", 0.0, api_key="sk-real-looking-key"))
    assert not has_live_credentials(ProviderConfig("custom", "m", 0.0))  # needs a base URL
    assert has_live_credentials(ProviderConfig("custom", "m", 0.0, base_url="http://localhost:8000/v1"))


def test_load_config_without_any_key(tmp_path: Path, clean_env) -> None:
    config = load_config(tmp_path)
    assert config.state_dir == tmp_path / "state" and config.state_dir.is_dir()
    assert config.data_dir == tmp_path / "data"
    assert config.compact_threshold_tokens == 1000 and config.compact_keep_messages == 4
    assert not has_live_credentials(config.model)
    assert config.judge_model.provider == config.model.provider


def test_load_config_reads_env_and_dotenv(tmp_path: Path, clean_env) -> None:
    (tmp_path / ".env").write_text(
        "# comment\nLLM_PROVIDER=anthorpic\nANTHROPIC_API_KEY=sk-ant-test\nCOMPACT_THRESHOLD_TOKENS=321\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.model.provider == "anthropic" and config.model.api_key == "sk-ant-test"
    assert config.compact_threshold_tokens == 321
    assert has_live_credentials(config.model)


def test_load_config_autodetects_provider_from_key(tmp_path: Path, clean_env) -> None:
    os.environ["GEMINI_API_KEY"] = "AIza-test"
    assert load_config(tmp_path).model.provider == "gemini"


def test_load_config_ignores_placeholder_key_from_readme(tmp_path: Path, clean_env) -> None:
    (tmp_path / ".env").write_text("LLM_PROVIDER=openai\nOPENAI_API_KEY=...\n", encoding="utf-8")
    config = load_config(tmp_path)
    assert not has_live_credentials(config.model)


def test_agents_fall_back_to_offline_without_credentials(tmp_path: Path, clean_env) -> None:
    config = load_config(tmp_path)  # no keys anywhere
    for agent_cls in (BaselineAgent, AdvancedAgent):
        agent = agent_cls(config)  # force_offline=False: auto-detect must land on offline, not crash
        assert agent.langchain_agent is None and agent.mode == "offline"
        result = agent.reply("u", "t", "Mình tên là DũngCT.")
        assert result["mode"] == "offline" and result["response"]


class _ExplodingLiveAgent:
    def invoke(self, *args, **kwargs):
        raise ConnectionError("network is down")


@pytest.mark.parametrize("agent_cls", [BaselineAgent, AdvancedAgent])
def test_live_failure_degrades_to_offline(tmp_path: Path, agent_cls) -> None:
    agent = agent_cls(make_config(tmp_path), force_offline=True)
    agent.langchain_agent = _ExplodingLiveAgent()  # simulate a configured-but-broken live model

    with pytest.warns(RuntimeWarning, match="falling back to offline"):
        first = agent.reply("u", "t", "Mình tên là DũngCT.")
    assert first["mode"] == "offline" and "network is down" in agent.live_error
    assert agent.langchain_agent is None  # stays offline; no repeated slow failures

    second = agent.reply("u", "t", "Tên mình là gì?")
    assert "DũngCT" in second["response"]  # the failed live turn was not lost or duplicated


# ---------------------------------------------------------------------------------------
# Live wiring, verified with a scripted fake chat model (no network, no API key)
# ---------------------------------------------------------------------------------------


def _scripted_model():
    pytest.importorskip("langchain")
    pytest.importorskip("langgraph")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ScriptedChatModel(BaseChatModel):
        """Calls `update_user_profile` when the user mentions Huế, otherwise answers in text."""

        calls: int = 0

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls += 1
            text = "\n".join(str(m.content) for m in messages)
            usage = {"input_tokens": 100, "output_tokens": 7, "total_tokens": 107}
            if "Messages to summarize" in text or "Context Extraction" in text:
                reply = AIMessage(content="TÓM TẮT: người dùng ở Huế.", usage_metadata=usage)
            elif isinstance(messages[-1], ToolMessage):
                reply = AIMessage(content="Đã lưu nơi ở của bạn.", usage_metadata=usage)
            elif "tool-please" in str(messages[-1].content):
                reply = AIMessage(
                    content="",
                    tool_calls=[{"name": "update_user_profile", "args": {"field": "pet", "value": "mèo Mun"}, "id": "c1"}],
                    usage_metadata=usage,
                )
            else:
                reply = AIMessage(content="Xin chào, mình đây.", usage_metadata=usage)
            return ChatResult(generations=[ChatGeneration(message=reply)])

    return ScriptedChatModel()


def _live_ready(monkeypatch, model) -> None:
    for module in (agent_baseline, agent_advanced):
        monkeypatch.setattr(module, "has_live_credentials", lambda _config: True)
        monkeypatch.setattr(module, "build_chat_model", lambda _config, m=model: m)


def test_baseline_live_wiring(tmp_path: Path, monkeypatch) -> None:
    _live_ready(monkeypatch, _scripted_model())
    agent = BaselineAgent(make_config(tmp_path))
    assert agent.mode == "live", agent.live_error

    result = agent.reply("u", "t1", "Chào bạn")
    assert result["mode"] == "live" and result["response"] == "Xin chào, mình đây."
    assert result["prompt_tokens"] == 100  # provider-reported usage is preferred over estimates
    assert agent.token_usage("t1") > 0 and agent.prompt_token_usage("t1") == 100


def test_advanced_live_wiring_tools_and_guardrail(tmp_path: Path, monkeypatch) -> None:
    _live_ready(monkeypatch, _scripted_model())
    agent = AdvancedAgent(make_config(tmp_path))
    assert agent.mode == "live", agent.live_error

    # The model decides to call the write tool -> User.md changes through the tool.
    result = agent.reply("u", "t1", "tool-please nhớ giúp mình")
    assert result["mode"] == "live" and result["response"] == "Đã lưu nơi ở của bạn."
    assert agent.profile_store.facts("u")["pet"] == "mèo Mun"

    # The rule-based guardrail persists clear facts even when the model never calls a tool.
    agent.reply("u", "t1", "Mình ở Huế và đang làm MLOps engineer.")
    facts = agent.profile_store.facts("u")
    assert facts["location"] == "Huế" and facts["profession"] == "MLOps engineer"
    assert agent.prompt_token_usage("t1") >= 200


def test_advanced_live_counts_summarization(tmp_path: Path, monkeypatch) -> None:
    _live_ready(monkeypatch, _scripted_model())
    agent = AdvancedAgent(make_config(tmp_path, threshold=60, keep=2))
    assert agent.mode == "live", agent.live_error

    for index in range(6):
        agent.reply("u", "long", f"Tin số {index}. " + "mình kể một đoạn khá dài. " * 8)
    assert agent.compaction_count("long") >= 1  # SummarizationMiddleware really compacted the thread
