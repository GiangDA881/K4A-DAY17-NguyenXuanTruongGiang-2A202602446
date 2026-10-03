from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------------------
# Token estimation
# --------------------------------------------------------------------------------------

# Shared by both agents so the prompt-cost comparison stays fair.
BASE_SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời bằng tiếng Việt. Hãy dùng ngữ cảnh được cung cấp "
    "(nếu có) để trả lời chính xác và nhất quán."
)


def estimate_tokens(text: str) -> int:
    """Deterministic token estimator: ~4 characters per token, 0 for empty text.

    Not a real tokenizer; it only needs to be stable so offline benchmarks are repeatable.
    """

    text = (text or "").strip()
    if not text:
        return 0
    return math.ceil(len(text) / 4)


SYSTEM_PROMPT_TOKENS = estimate_tokens(BASE_SYSTEM_PROMPT)


# --------------------------------------------------------------------------------------
# Profile schema
# --------------------------------------------------------------------------------------

# Scalar fields hold one current value: a newer value *replaces* the old one (conflict handling).
SCALAR_FIELDS = ("name", "location", "profession", "favorite_drink", "favorite_food", "pet")
# List fields hold a bounded, recency-ordered set of items (cheap memory decay).
LIST_FIELDS = ("interests", "hobbies", "style")
FIELD_ORDER = SCALAR_FIELDS + LIST_FIELDS

FIELD_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "interests": "Mối quan tâm",
    "hobbies": "Sở thích",
    "style": "Style trả lời",
}

MAX_LIST_ITEMS = {"interests": 8, "hobbies": 6, "style": 6}

# Confidence threshold: candidates below this are never written to User.md.
MIN_CONFIDENCE = 0.7


@dataclass(frozen=True)
class FactCandidate:
    field: str
    value: str
    confidence: float


# --------------------------------------------------------------------------------------
# Fact merging (shared by User.md and the baseline's in-thread scratch facts)
# --------------------------------------------------------------------------------------


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _style_key(descriptor: str) -> str:
    """Canonical key so that '3 bullet' replaces 'bullet' instead of piling up duplicates."""

    low = descriptor.lower()
    if "bullet" in low:
        return "bullets"
    if "ngắn" in low or "gọn" in low:
        return "concise"
    if "ví dụ" in low:
        return "examples"
    if "trade-off" in low:
        return "tradeoff"
    if "cấu trúc" in low:
        return "structured"
    return low


def _item_key(field_name: str, item: str) -> str:
    return _style_key(item) if field_name == "style" else item.casefold()


def _is_less_specific(new: str, old: str) -> bool:
    """True when `new` only restates part of `old` ("bullet" vs "3 bullet", "có ví dụ" vs "ví dụ thực tế")."""

    def core(text: str) -> str:
        return re.sub(r"^có\s+", "", text.casefold())

    return core(new) != core(old) and core(new) in core(old)


def merge_fact(facts: dict[str, str], field_name: str, value: str) -> str:
    """Merge one extracted value into `facts` in place.

    Returns the *delta* that was actually added/changed ("" when nothing semantically changed),
    which callers use for acknowledgements and to decide whether to touch the file.

    - scalar fields: a different value replaces the old one, so a correction never leaves the
      stale fact behind. A strictly less specific value ("MLOps" vs "MLOps engineer") is ignored.
    - list fields: items are deduplicated, re-mentions move to the end (recency), and the list is
      capped so the profile cannot grow without bound.
    """

    value = value.strip()
    if not value:
        return ""

    if field_name in LIST_FIELDS:
        items = _split_list(facts.get(field_name, ""))
        key = _item_key(field_name, value)
        existing = next((i for i, item in enumerate(items) if _item_key(field_name, item) == key), None)
        delta = ""
        if existing is None:
            delta = value
        else:
            old_item = items.pop(existing)
            if _is_less_specific(value, old_item):
                value = old_item  # a vaguer re-mention ("bullet") must not erase "3 bullet"
            elif old_item != value:
                delta = value  # same concept, more specific/newer wording
        items.append(value)
        cap = MAX_LIST_ITEMS.get(field_name, 8)
        facts[field_name] = ", ".join(items[-cap:])
        return delta

    old = facts.get(field_name)
    if old is not None:
        if old.casefold() == value.casefold():
            return ""
        if value.casefold() in old.casefold():
            return ""  # keep the more specific value we already have
    facts[field_name] = value
    return value


def apply_updates_to_facts(facts: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    """Merge `updates` (field -> value, list fields comma-joined) into `facts`; return the deltas."""

    changes: dict[str, str] = {}
    for field_name, raw_value in updates.items():
        values = _split_list(raw_value) if field_name in LIST_FIELDS else [raw_value]
        deltas = [d for d in (merge_fact(facts, field_name, v) for v in values) if d]
        if deltas:
            changes[field_name] = ", ".join(deltas)
    return changes


# --------------------------------------------------------------------------------------
# User.md persistence
# --------------------------------------------------------------------------------------

_FACT_LINE = re.compile(r"^-\s+([A-Za-z_][\w]*)\s*:\s*(.+?)\s*$")
_DEFAULT_PROFILE = "# User Profile\n\n## Facts\n"


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`, one markdown file per user.

    Layout: `<root_dir>/<slug>/User.md` (e.g. `state/profiles/dungct/User.md`).
    Facts are plain `- key: value` lines so the file stays human-readable and hand-editable.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        raw = (user_id or "").strip()
        slug = re.sub(r"[^\w\-]+", "_", raw).strip("._") or "anonymous"
        if slug != raw:
            # Sanitising can make two ids collide ("a b" vs "a_b"); a short hash keeps them apart.
            slug = f"{slug}-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:6]}"
        return Path(self.root_dir) / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.is_file():
            return _DEFAULT_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash can never leave a half-written profile behind.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".User-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence of `search_text`; return whether the file changed."""

        if not search_text:
            return False
        text = self.read_text(user_id)
        if search_text not in text:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.is_file() else 0

    # ---- structured helpers -------------------------------------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = _FACT_LINE.match(line)
            if match:
                found[match.group(1)] = match.group(2)
        return found

    def upsert_fact(self, user_id: str, field_name: str, value: str) -> bool:
        """Insert/merge one fact into User.md. Returns True when something changed."""

        return bool(self.apply_updates(user_id, {field_name: value}))

    def apply_updates(self, user_id: str, updates: dict[str, str]) -> dict[str, str]:
        """Merge `updates` into User.md, rewriting only the affected `- key:` lines.

        Everything else in the file (headings, hand-written notes) is preserved. The file is only
        written when its content really changes. Returns the deltas (see `merge_fact`).
        """

        if not updates:
            return {}
        current = self.facts(user_id)
        merged = dict(current)
        changes = apply_updates_to_facts(merged, updates)

        text = self.read_text(user_id)
        lines = text.splitlines()
        written: set[str] = set()
        for index, line in enumerate(lines):
            match = _FACT_LINE.match(line)
            if match and match.group(1) in merged:
                key = match.group(1)
                lines[index] = f"- {key}: {merged[key]}"
                written.add(key)

        missing = [key for key in merged if key not in written]
        if missing:
            insert_at = max((i for i, ln in enumerate(lines) if _FACT_LINE.match(ln)), default=-1) + 1
            if insert_at == 0:
                header = next((i for i, ln in enumerate(lines) if ln.strip() == "## Facts"), None)
                insert_at = header + 1 if header is not None else len(lines)
            ordered = [k for k in FIELD_ORDER if k in missing] + [k for k in missing if k not in FIELD_ORDER]
            lines[insert_at:insert_at] = [f"- {key}: {merged[key]}" for key in ordered]

        new_text = "\n".join(lines).rstrip("\n") + "\n"
        if new_text != text:
            self.write_text(user_id, new_text)
        return changes


# --------------------------------------------------------------------------------------
# Fact extraction (rule-based, offline)
# --------------------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"\s*,\s*|\s+nhưng\s+")

_QUESTION_RE = re.compile(
    r"\?\s*$|\b(?:là gì|là ai|ở đâu|tên gì|làm gì|nghề gì|con gì|món gì|thế nào)\b", re.IGNORECASE
)
# Sentences that are jokes / hypotheticals are never durable facts.
_HYPOTHETICAL_MARKERS = ("đùa", "hay là", "giả sử", "giả dụ", "tưởng tượng", "nếu mình", "nếu tôi", "nếu như mình")
# Clauses describing the *old* state of a correction ("Lúc đầu mình nói hiện ở Huế, nhưng ...").
_PAST_MARKERS = ("lúc đầu", "ban đầu", "trước đó", "trước đây", "trước kia", "hồi trước", "hồi đó")
_TEMPORARY_MARKERS = ("tạm thời",)

_ADVERBS = r"(?:(?:đang|hiện\s+tại|hiện|vẫn|còn|giờ|đã)\s+)*"

_NAME_PREFIXES = [
    (re.compile(r"(?i:\b(?:mình|tôi|tớ)\s+tên(?:\s+là)?\s+)"), 0.95),
    (re.compile(r"(?i:\btên\s+(?:của\s+)?(?:mình|tôi)\s+là\s+)"), 0.95),
    (re.compile(r"(?i:\b(?:gọi|kêu)\s+(?:mình|tôi)\s+là\s+)"), 0.85),
]

_LOCATION_PREFIXES = [
    (re.compile(rf"(?i:\b(?:mình|tôi)\s+{_ADVERBS}ở\s+)"), 0.9),
    (re.compile(rf"(?i:\b(?:mình|tôi)\s+{_ADVERBS}(?:làm\s+việc|sống|sinh\s+sống)\s+ở\s+)"), 0.8),
    (re.compile(r"(?i:\bhiện\s+(?:tại\s+)?(?:đang\s+)?(?:ở|sống\s+ở)\s+)"), 0.85),
    # Subject-dropping coordination: "Mình đang làm MLOps engineer và đang ở Đà Nẵng".
    # Requiring the conjunction avoids "đồng nghiệp đang ở Singapore" (someone else).
    (re.compile(r"(?i:\bvà\s+(?:hiện\s+)?đang\s+ở\s+)"), 0.8),
    (re.compile(r"(?i:\bnơi\s+ở(?:\s+hiện\s+tại)?(?:\s+của\s+mình)?\s+là\s+)"), 0.9),
    (re.compile(r"(?i:\b(?:chuyển|dọn)\s+(?:đến|về|ra|vào)\s+)"), 0.85),
]

_PROFESSION_PREFIXES = [
    (re.compile(rf"(?i:\b(?:mình|tôi)\s+{_ADVERBS}làm\s+(?:một\s+)?)"), 0.9),
    (re.compile(r"(?i:\b(?:đang|hiện(?:\s+tại)?)\s+làm\s+(?:một\s+)?)"), 0.9),
    (re.compile(r"(?i:\bnghề(?:\s+nghiệp)?(?:\s+hiện\s+tại|\s+mới)?(?:\s+của\s+mình)?(?:\s+thì)?(?:\s+vẫn)?\s+là\s+)"), 0.9),
    (re.compile(r"(?i:\b(?:công\s+việc|vị\s+trí)(?:\s+hiện\s+tại)?(?:\s+của\s+mình)?\s+là\s+)"), 0.85),
    (re.compile(r"(?i:\bchuyển\s+sang\s+)"), 0.85),
    (re.compile(r"(?i:\bnghề\s+)"), 0.8),
]

_DRINK_PREFIXES = [
    (re.compile(r"(?i:\bđồ\s+uống\s+(?:yêu|ưa)\s+thích(?:\s+của\s+mình)?\s+là\s+)"), 0.95),
    # Habit, not preference: stays below MIN_CONFIDENCE so it is never persisted on its own.
    (re.compile(r"(?i:(?<!đồ\s)\b(?:(?:vẫn|hay|thường|luôn)\s+)*uống\s+)"), 0.65),
]
_FOOD_PREFIXES = [
    (re.compile(r"(?i:\bmón(?:\s+ăn)?\s+(?:yêu|ưa)\s+thích(?:\s+của\s+mình)?\s+là\s+)"), 0.95),
]

# `(?<!giải\s)` keeps "giải thích benchmark" (explain) from being read as "thích benchmark" (like);
# `(?<!không\s)` skips negated preferences ("không thích câu trả lời lan man").
_LIST_PREFIX = re.compile(
    r"(?i:(?<!không\s)(?<!giải\s)\b(?:thích|quan\s+tâm(?:\s+nhiều)?(?:\s+(?:đến|tới|về))?|đam\s+mê)\s+)"
)
_PET_NUOI = re.compile(
    r"(?i:\bnuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?)([^\W\d_][\w\-]*)(?:\s+(?i:tên)\s+([^\W\d_][\w\-]*))?"
)
_PET_CON = re.compile(r"(?i:\b(?:con|bé)\s+)([^\W\d_][\w\-]*)\s+(?i:tên)\s+([^\W\d_][\w\-]*)")
_FROM_TO = re.compile(r"(?i:\b(?:chuyển|đổi|cập\s+nhật|dời)\s+từ\s+)")

_JOB_ROLES = {
    "engineer", "developer", "dev", "manager", "scientist", "analyst", "designer", "architect", "lead",
    "specialist", "researcher", "consultant", "devops", "mlops", "sre", "admin", "administrator",
    "tester", "qa", "intern", "founder", "ceo", "cto", "student", "teacher", "programmer", "freelancer",
}
_VN_TITLES = (
    "kỹ sư", "lập trình viên", "giáo viên", "giảng viên", "bác sĩ", "chuyên viên",
    "nhà phát triển", "nhà thiết kế", "sinh viên",
)
_STOP_WORDS = {
    "cho", "tại", "ở", "và", "với", "của", "để", "từ", "nữa", "nhé", "chứ", "thì", "nên", "vì",
    "nhưng", "rồi", "khi", "hiện", "giờ", "đã", "vẫn", "không", "chưa", "mới", "cũ",
}
_ASCII_WORD = re.compile(r"\s*([A-Za-z][A-Za-z0-9\-]*)(?!\w)")
_ANY_WORD = re.compile(r"\s*([^\W\d_][\w\-]*)")
_PHRASE_END = re.compile(r"[,.;:!?]|\s+(?:và|nhưng|vì|chứ|như|để|nhé|nha|rồi|khi|nên|thì)\b")

_DRINK_WORDS = ("cà phê", "trà", "bia", "rượu", "nước", "sinh tố", "sữa", "matcha", "cacao", "coffee")
_FOOD_WORDS = ("mì quảng", "phở", "bún", "cơm", "bánh", "chè", "gà", "lẩu", "hủ tiếu", "pizza")
_TECH_WORDS = {
    "python", "ai", "ml", "mlops", "devops", "rag", "llm", "agent", "agents", "benchmark", "memory",
    "async", "backend", "frontend", "data", "nlp", "api", "cloud", "docker", "kubernetes", "evaluation",
    "code", "coding", "java", "javascript", "typescript", "rust", "golang", "sql",
}
_HOBBY_VERBS = ("chạy", "nghe", "đọc", "xem", "đi", "chơi", "chụp", "nấu", "bơi", "tập", "vẽ", "leo")
_NOT_AN_ITEM = ("cách", "kiểu", "câu", "việc", "những", "cái", "được", "là ", "của")

_STYLE_CUE = re.compile(r"(?i:trả lời|giải thích|trình bày|style|phong cách|giữ câu|bullet)")
_STYLE_INSTRUCTION = re.compile(r"(?i:muốn|thích|hãy|nên|ưu tiên|đừng|giữ|bạn thử|nhớ|theo)")
_STYLE_DESCRIPTORS = [
    (re.compile(r"(?i:\b(\d+)\s*bullet)"), lambda m: f"{m.group(1)} bullet"),
    (re.compile(r"(?i:\bbullet)"), lambda m: "bullet"),
    (re.compile(r"(?i:ngắn gọn|\bgọn\b|\bngắn\b|lan man)"), lambda m: "ngắn gọn"),
    (re.compile(r"(?i:ví dụ thực chiến)"), lambda m: "ví dụ thực chiến"),
    (re.compile(r"(?i:ví dụ thực tế)"), lambda m: "ví dụ thực tế"),
    (re.compile(r"(?i:\bví dụ\b)"), lambda m: "có ví dụ"),
    (re.compile(r"(?i:trade-?off)"), lambda m: "nhấn trade-off"),
    (re.compile(r"(?i:có cấu trúc)"), lambda m: "có cấu trúc"),
]


def _leading_capitalized(text: str, max_words: int = 4) -> str:
    """Consume leading Capitalized words ("Đà Nẵng", "DũngCT Stress"); stop at punctuation."""

    words: list[str] = []
    pos = 0
    while len(words) < max_words:
        match = _ANY_WORD.match(text, pos)
        if not match or not match.group(1)[0].isupper():
            break
        words.append(match.group(1))
        pos = match.end()
        if pos < len(text) and text[pos] in ",.;:!?)":
            break
    return " ".join(words)


def _leading_job(text: str) -> str:
    """Parse a job title ("MLOps engineer", "kỹ sư dữ liệu") and require a recognisable role word."""

    text = text.lstrip()
    low = text.lower()
    for title in _VN_TITLES:
        if low.startswith(title) and (len(low) == len(title) or not low[len(title)].isalnum()):
            words = [text[: len(title)]]
            pos = len(title)
            while len(words) < 4:
                match = _ANY_WORD.match(text, pos)
                if not match or match.group(1).lower() in _STOP_WORDS:
                    break
                words.append(match.group(1))
                pos = match.end()
            return " ".join(words)

    words = []
    pos = 0
    while len(words) < 4:
        match = _ASCII_WORD.match(text, pos)
        if not match or match.group(1).lower() in _STOP_WORDS:
            break
        words.append(match.group(1))
        pos = match.end()
        if pos < len(text) and text[pos] in ",.;:!?)":
            break
    if words and words[-1].lower() in _JOB_ROLES:
        return " ".join(words)
    return ""


def _leading_phrase(text: str, max_words: int = 5) -> str:
    text = text.strip()
    end = _PHRASE_END.search(text)
    phrase = text[: end.start()] if end else text
    return " ".join(phrase.split()[:max_words]).strip()


def _split_items(tail: str) -> list[str]:
    tail = re.split(r"\s+(?:vì|nhưng|để|do|khi|rồi|nên|chứ)\s+", tail, maxsplit=1)[0]
    tail = tail.strip().rstrip(".!?;")
    parts = re.split(r"\s*,\s*|\s+và\s+|\s+cùng\s+", tail)
    return [re.sub(r"^(?:cả|cũng|những)\s+", "", part.strip()) for part in parts if part.strip()]


def _classify_item(item: str) -> tuple[str, str] | None:
    """Map a 'thích/quan tâm' list item to a profile field, or None if it is not a stable fact."""

    low = item.lower()
    words = low.split()
    if not words or len(words) > 5 or low.startswith(_NOT_AN_ITEM):
        return None
    if any(word in low for word in _DRINK_WORDS):
        return "favorite_drink", item
    if any(word in low for word in _FOOD_WORDS):
        return "favorite_food", item
    if _TECH_WORDS.intersection(re.findall(r"\w+", low)):
        return "interests", item
    if words[0] in _HOBBY_VERBS:
        return "hobbies", item
    return None


def _prepare_sentence(sentence: str) -> str | None:
    """Filter a sentence down to the text that may contain durable facts, or None to skip it."""

    sentence = sentence.strip()
    if not sentence or _QUESTION_RE.search(sentence):
        return None  # a question is a request, not a fact (avoids storing "tên mình là gì")
    low = sentence.lower()
    if any(marker in low for marker in _HYPOTHETICAL_MARKERS + _TEMPORARY_MARKERS):
        return None
    clauses = [c for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]
    kept = [c for c in clauses if not any(marker in c.lower() for marker in _PAST_MARKERS)]
    return ", ".join(kept) if kept else None


def _sentence_candidates(sentence: str) -> list[tuple[int, FactCandidate]]:
    found: list[tuple[int, FactCandidate]] = []

    def add(position: int, field_name: str, value: str, confidence: float) -> None:
        value = value.strip(" .,;:!?\"'")
        if value and value.lower() not in {"gì", "gì đó", "ai"}:
            found.append((position, FactCandidate(field_name, value, confidence)))

    for regex, confidence in _NAME_PREFIXES:
        for m in regex.finditer(sentence):
            add(m.start(), "name", _leading_capitalized(sentence[m.end():]), confidence)

    for regex, confidence in _LOCATION_PREFIXES:
        for m in regex.finditer(sentence):
            add(m.start(), "location", _leading_capitalized(sentence[m.end():]), confidence)

    for regex, confidence in _PROFESSION_PREFIXES:
        for m in regex.finditer(sentence):
            add(m.start(), "profession", _leading_job(sentence[m.end():]), confidence)

    for regex, confidence in _DRINK_PREFIXES:
        for m in regex.finditer(sentence):
            value = _leading_phrase(sentence[m.end():])
            is_drink_word = any(word in value.lower() for word in _DRINK_WORDS)
            if confidence >= 0.9 or is_drink_word:
                add(m.start(), "favorite_drink", value, confidence)

    for regex, confidence in _FOOD_PREFIXES:
        for m in regex.finditer(sentence):
            add(m.start(), "favorite_food", _leading_phrase(sentence[m.end():]), confidence)

    for m in _PET_NUOI.finditer(sentence):
        species, pet_name = m.group(1), m.group(2)
        if species.lower() not in {"gì", "nhiều", "thêm"} and not species[0].isupper():
            add(m.start(), "pet", f"{species} tên {pet_name}" if pet_name else species, 0.9)
    for m in _PET_CON.finditer(sentence):
        if m.group(2)[0].isupper():
            add(m.start(), "pet", f"{m.group(1)} tên {m.group(2)}", 0.85)

    for m in _FROM_TO.finditer(sentence):
        old, _, rest = sentence[m.end():].partition(" sang ")
        low = sentence.lower()
        new_place = _leading_capitalized(rest)
        if new_place and _leading_capitalized(old) and ("nơi ở" in low or re.search(r"\bở\b", low)):
            add(m.start(), "location", new_place, 0.9)
        elif ("nghề" in low or "công việc" in low or "làm" in low) and rest:
            add(m.start(), "profession", _leading_job(rest), 0.85)

    for m in _LIST_PREFIX.finditer(sentence):
        tail = sentence[m.end():]
        if tail.lower().startswith("là "):
            continue  # "đồ uống yêu thích là ..." is handled by the explicit patterns above
        for item in _split_items(tail):
            classified = _classify_item(item)
            if classified:
                add(m.start(), classified[0], classified[1], 0.85 if classified[0] != "hobbies" else 0.8)

    if _STYLE_CUE.search(sentence):
        explicit = bool(_STYLE_INSTRUCTION.search(sentence))
        seen_keys: set[str] = set()
        for regex, build in _STYLE_DESCRIPTORS:
            match = regex.search(sentence)
            if not match:
                continue
            descriptor = build(match)
            key = _style_key(descriptor)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            add(match.start(), "style", descriptor, 0.85 if explicit else 0.6)

    return found


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """All fact candidates found in `message`, with confidence, in textual order.

    Question sentences, jokes/hypotheticals, "temporary" statements and the *old* side of a
    correction are skipped, so "Hà Nội chỉ là nơi mình vừa đi họp" or "hay là chuyển sang product
    manager ... chỉ là câu đùa" never become facts.
    """

    candidates: list[FactCandidate] = []
    for raw_sentence in _SENTENCE_SPLIT.split(message or ""):
        sentence = _prepare_sentence(raw_sentence)
        if sentence is None:
            continue
        ordered = sorted(_sentence_candidates(sentence), key=lambda pair: pair[0])
        candidates.extend(candidate for _, candidate in ordered)
    return candidates


def extract_profile_updates(message: str, min_confidence: float = MIN_CONFIDENCE) -> dict[str, str]:
    """Convert raw user text into stable profile facts (field -> value).

    Scalar fields keep the single best candidate (highest confidence, latest wins ties); list fields
    (interests, hobbies, style) return their items comma-joined. Candidates below `min_confidence`
    are dropped (confidence threshold), so weak signals like "mình hay uống ..." are not persisted.
    """

    scalar_best: dict[str, tuple[float, int, str]] = {}
    list_items: dict[str, list[str]] = {}
    for index, candidate in enumerate(extract_profile_candidates(message)):
        if candidate.confidence < min_confidence:
            continue
        if candidate.field in LIST_FIELDS:
            bucket = list_items.setdefault(candidate.field, [])
            if candidate.value not in bucket:
                bucket.append(candidate.value)
        else:
            rank = (candidate.confidence, index, candidate.value)
            if candidate.field not in scalar_best or rank[:2] >= scalar_best[candidate.field][:2]:
                scalar_best[candidate.field] = rank

    updates = {name: entry[2] for name, entry in scalar_best.items()}
    updates.update({name: ", ".join(items) for name, items in list_items.items()})
    return {name: updates[name] for name in FIELD_ORDER if name in updates}


# --------------------------------------------------------------------------------------
# Offline answer composition (shared so baseline and advanced differ ONLY in memory)
# --------------------------------------------------------------------------------------

_INTENTS = [
    ("name", re.compile(r"(?i:\btên\b|\blà ai\b)")),
    ("location", re.compile(r"(?i:nơi ở|ở đâu|đang ở|còn ở|hiện ở|sống ở)")),
    ("profession", re.compile(r"(?i:\bnghề\b|làm gì|công việc|chức danh)")),
    ("favorite_drink", re.compile(r"(?i:đồ uống|uống gì|thức uống)")),
    ("favorite_food", re.compile(r"(?i:món ăn|ăn gì)")),
    ("pet", re.compile(r"(?i:nuôi|thú cưng|con gì)")),
    ("style", re.compile(r"(?i:\bstyle\b|kiểu trả lời|phong cách|trả lời như thế nào|cách trả lời)")),
    ("interests", re.compile(r"(?i:quan tâm|kỹ thuật chính|chủ đề)")),
    ("hobbies", re.compile(r"(?i:sở thích|thích gì)")),
]
_RECALL_CUE = re.compile(r"(?i:nhắc lại|nhớ lại|là gì|tóm tắt|cho mình biết|còn nhớ)")


def is_recall_question(message: str) -> bool:
    return "?" in message or bool(_RECALL_CUE.search(message))


def detect_intents(message: str) -> list[str]:
    return [name for name, regex in _INTENTS if regex.search(message)]


def render_recall_answer(message: str, facts: dict[str, str], missing_scope: str = "bộ nhớ") -> str:
    """Answer a recall question from `facts` only: short bullets, no echo of the question."""

    intents = detect_intents(message) or [name for name in FIELD_ORDER if name in facts]
    lines: list[str] = []
    missing: list[str] = []
    for name in intents:
        value = facts.get(name)
        if not value:
            missing.append(FIELD_LABELS[name].lower())
            continue
        if name == "interests":
            value = ", ".join(_split_list(value)[-4:])
        lines.append(f"- {FIELD_LABELS[name]}: {value}")
    if not lines:
        what = ", ".join(missing) if missing else "bạn"
        return f"Mình chưa có thông tin về {what} trong {missing_scope}."
    if missing:
        lines.append(f"- Chưa có thông tin: {', '.join(missing)}.")
    return "\n".join(lines)


def render_ack(changes: dict[str, str], destination: str | None = None) -> str:
    """Short acknowledgement for a non-question turn."""

    if not changes or not destination:
        return "Đã ghi nhận."
    saved = "; ".join(f"{FIELD_LABELS[name].lower()} = {value}" for name, value in changes.items())
    return f"Đã lưu vào {destination}: {saved}."


# --------------------------------------------------------------------------------------
# Compact memory
# --------------------------------------------------------------------------------------


def _snippet(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    if len(first) > limit:
        first = first[:limit].rsplit(" ", 1)[0] + "…"
    return first


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages: one short line per message, newest `max_items` kept.

    Durable facts are NOT the summary's job (they already live in User.md); the summary only keeps
    the gist of the recent-but-evicted conversation. Swap in an LLM summariser for live mode.
    """

    if not messages:
        return ""
    chosen = messages[-max_items:]
    lines: list[str] = []
    omitted = len(messages) - len(chosen)
    if omitted:
        lines.append(f"(+{omitted} tin nhắn cũ hơn đã lược)")
    for message in chosen:
        is_user = message.get("role") == "user"
        lines.append(f"- {'User' if is_user else 'Agent'}: {_snippet(message.get('content', ''), 100 if is_user else 60)}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    - keeps the most recent `keep_messages` messages verbatim
    - when summary + messages exceed `threshold_tokens`, older messages are folded into `summary`
    - the summary itself is capped (`max_summary_tokens`) so it cannot grow without bound
    - counts compactions per thread for the benchmark
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)
    summary_items: int = 6
    max_summary_tokens: int = 220

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.context(thread_id)
        thread["messages"].append({"role": role, "content": content})  # type: ignore[union-attr]
        if self.context_tokens(thread_id) > self.threshold_tokens:
            self._compact(thread)

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])  # type: ignore[arg-type]

    def context_tokens(self, thread_id: str) -> int:
        """Tokens this thread would carry into the next prompt: summary + recent messages."""

        thread = self.context(thread_id)
        messages = thread["messages"]
        return estimate_tokens(str(thread["summary"])) + sum(
            estimate_tokens(m["content"]) for m in messages  # type: ignore[union-attr]
        )

    def _compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        if len(messages) <= self.keep_messages:
            return  # nothing older than the protected tail; avoids looping on one huge message
        older, recent = messages[: -self.keep_messages], messages[-self.keep_messages :]
        combined = f"{thread['summary']}\n{summarize_messages(older, self.summary_items)}"
        lines = [line for line in combined.splitlines() if line.strip()]
        while len(lines) > 1 and estimate_tokens("\n".join(lines)) > self.max_summary_tokens:
            lines.pop(0)  # forget the oldest summary lines first
        thread["summary"] = "\n".join(lines)
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[arg-type]
