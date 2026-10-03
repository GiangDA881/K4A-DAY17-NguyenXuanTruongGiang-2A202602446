from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from model_provider import build_chat_model, has_live_credentials, message_text

COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""

    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else [data]


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _contains(answer: str, expected: str) -> bool:
    """Whole-word match. Acronyms such as "AI" are case-sensitive so they cannot match "hai"/"bài"."""

    answer, expected = _nfc(answer), _nfc(expected).strip()
    if not expected:
        return False
    flags = 0 if (expected.isascii() and expected.isupper()) else re.IGNORECASE
    return re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", answer, flags) is not None


def recall_points(answer: str, expected: list[str]) -> float:
    """Fraction of expected facts present in the answer (0 / 0.5 / 1 for two facts)."""

    if not expected:
        return 0.0
    return sum(_contains(answer, item) for item in expected) / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight offline quality score in [0, 1].

    80% of the weight is correctness (recall of the expected facts) scaled by 20% conciseness:
    answers up to 300 characters get full form credit, decaying to 0 at 900. An answer that
    recalls nothing scores 0 however short it is, so an honest "I don't know" is not rewarded.
    """

    recall = recall_points(answer, expected)
    length = len(answer.strip())
    conciseness = 1.0 if length <= 300 else max(0.0, 1.0 - (length - 300) / 600)
    return round(recall * (0.8 + 0.2 * conciseness), 4)


def llm_judge_quality(config, question: str, answer: str, expected: list[str]) -> float | None:
    """Optional live-mode judge using `config.judge_model`. Returns None when unavailable."""

    if not has_live_credentials(config.judge_model):
        return None
    prompt = (
        "Chấm câu trả lời của một trợ lý AI từ 0 đến 1 (1 = đúng và đủ các thông tin kỳ vọng, "
        "ngắn gọn; 0 = sai hoặc không biết).\n"
        f"Câu hỏi: {question}\nThông tin kỳ vọng: {', '.join(expected)}\nCâu trả lời: {answer}\n"
        "Chỉ trả lời bằng một số thực."
    )
    try:
        reply = build_chat_model(config.judge_model).invoke(prompt)
        match = re.search(r"\d+(?:\.\d+)?", message_text(reply.content))
        return min(1.0, max(0.0, float(match.group()))) if match else None
    except Exception:
        return None


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config, use_judge: bool = False
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    1. feed every turn of a conversation to the agent in its own thread
    2. accumulate `agent tokens only` and `prompt tokens processed` from those ingest threads
       (recall questions are excluded so the cost columns measure the conversation itself)
    3. ask each recall question in a FRESH thread, so only persistent memory can answer it
    4. average recall and quality
    5. record User.md growth and compaction count

    Profiles persist across conversations of the same user, like real repeated sessions.
    """

    user_ids = sorted({conv["user_id"] for conv in conversations})
    size_before = sum(agent.memory_file_size(user_id) for user_id in user_ids)
    agent_tokens = prompt_tokens = compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        user_id, thread_id = conv["user_id"], conv["id"]
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)
        agent_tokens += agent.token_usage(thread_id)
        prompt_tokens += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

        for index, item in enumerate(conv.get("recall_questions", [])):
            answer = agent.reply(user_id, f"{thread_id}::recall::{index}", item["question"])["response"]
            expected = item["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            judged = llm_judge_quality(config, item["question"], answer, expected) if use_judge else None
            quality_scores.append(judged if judged is not None else heuristic_quality(answer, expected))

    size_after = sum(agent.memory_file_size(user_id) for user_id in user_ids)
    count = max(1, len(recall_scores))
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / count,
        response_quality=sum(quality_scores) / count,
        memory_growth_bytes=size_after - size_before,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render the rows as a markdown table (uses `tabulate` when installed)."""

    table = [
        [
            row.agent_name,
            f"{row.agent_tokens_only:,}",
            f"{row.prompt_tokens_processed:,}",
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            f"{row.memory_growth_bytes:,}",
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github", disable_numparse=True)
    except ImportError:
        lines = [
            "| " + " | ".join(COLUMNS) + " |",
            "|" + "|".join("---" for _ in COLUMNS) + "|",
        ]
        lines += ["| " + " | ".join(str(cell) for cell in line) + " |" for line in table]
        return "\n".join(lines)


def describe_comparison(baseline: BenchmarkRow, advanced: BenchmarkRow) -> list[str]:
    """Plain-language reading of one table, computed from the numbers (not hard-coded)."""

    notes = [
        f"Recall: Baseline {baseline.recall_score:.2f} -> Advanced {advanced.recall_score:.2f}"
        " (persistent User.md is what survives a new thread)."
    ]
    if baseline.prompt_tokens_processed:
        delta = (advanced.prompt_tokens_processed - baseline.prompt_tokens_processed) / baseline.prompt_tokens_processed
        direction = "fewer" if delta < 0 else "more"
        notes.append(
            f"Prompt tokens: Advanced processed {abs(delta):.0%} {direction} than Baseline"
            f" ({advanced.compactions} compaction(s))."
        )
    if baseline.agent_tokens_only:
        delta = (advanced.agent_tokens_only - baseline.agent_tokens_only) / baseline.agent_tokens_only
        notes.append(f"Agent tokens only: {delta:+.0%} (memory-write calls and richer replies).")
    notes.append(f"Memory growth: Advanced wrote {advanced.memory_growth_bytes:,} bytes of User.md.")
    return notes


SUITES = [
    ("standard", "Standard Benchmark", "conversations.json"),
    ("stress", "Long-Context Stress Benchmark", "advanced_long_context.json"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Baseline vs Advanced memory agents.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="use a real LLM when valid API credentials exist (default: deterministic offline mode)",
    )
    parser.add_argument("--suite", choices=["all", "standard", "stress"], default="all")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run both benchmark suites and print one comparison table per suite.

    Offline mode is the default so results are repeatable and need no API key. `--live` opts in to
    a real model, but still degrades to offline automatically when no valid credentials exist.
    """

    args = parse_args(argv)
    config = load_config(Path(__file__).resolve().parent.parent)

    for key, title, filename in SUITES:
        if args.suite not in ("all", key):
            continue

        # Isolated, freshly wiped state per suite: reruns never inherit an old User.md.
        suite_config = replace(config, state_dir=config.state_dir / "benchmark" / key)
        shutil.rmtree(suite_config.state_dir, ignore_errors=True)
        suite_config.state_dir.mkdir(parents=True, exist_ok=True)

        conversations = load_conversations(config.data_dir / filename)
        force_offline = not args.live
        baseline = BaselineAgent(suite_config, force_offline=force_offline)
        advanced = AdvancedAgent(suite_config, force_offline=force_offline)
        if args.live and (baseline.mode == "offline" or advanced.mode == "offline"):
            reason = advanced.live_error or "no valid API credentials found"
            print(f"[note] --live requested but running offline: {reason}")

        rows = [
            run_agent_benchmark("Baseline", baseline, conversations, suite_config, use_judge=args.live),
            run_agent_benchmark("Advanced", advanced, conversations, suite_config, use_judge=args.live),
        ]

        turns = sum(len(conv["turns"]) for conv in conversations)
        print(f"\n## {title}")
        print(f"{len(conversations)} conversation(s), {turns} turns, mode: {advanced.mode}\n")
        print(format_rows(rows))
        print()
        for note in describe_comparison(*rows):
            print(f"- {note}")


if __name__ == "__main__":
    main()
