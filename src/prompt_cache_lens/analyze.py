"""Compare consecutive requests, name what broke the prefix, and estimate what it cost."""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from . import facts
from .render import Rendered, common_prefix_length, render

CITE = "shared/prompt-caching.md"


@dataclass(frozen=True)
class Cause:
    id: str
    summary: str
    fix: str
    counts_as_waste: bool = True


_CAUSES = (
    Cause(
        "timestamp",
        "a timestamp changed inside the prefix",
        "Keep the prefix frozen. Move the time to the end of `messages`, or to a mid-conversation "
        f"system message where the model supports it. ({CITE}, Silent invalidators)",
    ),
    Cause(
        "request_id",
        "a per-request identifier changed inside the prefix",
        "Remove the ID from the prompt, or move it after the last breakpoint. Log it instead. "
        f"({CITE}, Silent invalidators)",
    ),
    Cause(
        "unsorted_tools",
        "the same tools arrived in a different order",
        f"Serialize tools deterministically: sort them by name. ({CITE}, Silent invalidators)",
    ),
    Cause(
        "json_key_order",
        "the same content was serialized with its keys in a different order",
        "Serialize with sorted keys, for example `json.dumps(d, sort_keys=True)`, and do not "
        f"build objects from sets. ({CITE}, Silent invalidators)",
    ),
    Cause(
        "tool_set_changed",
        "a tool was added or removed",
        "Keep the tool list fixed for a session. Use `defer_loading` or mid-conversation tool "
        f"changes where supported. ({CITE}, Invalidation hierarchy)",
    ),
    Cause(
        "tool_edited",
        "a tool definition changed",
        f"Freeze tool definitions for a session. ({CITE}, Invalidation hierarchy)",
    ),
    Cause(
        "system_edited",
        "the system prompt text changed",
        "Move changing sections out of top-level `system`, after the cached prefix. "
        f"({CITE}, Invalidation hierarchy)",
    ),
    Cause(
        "history_rewritten",
        "an earlier message changed",
        "Keep history append-only. Rewriting or summarising past turns invalidates everything "
        f"after the edit. ({CITE}, Finding the invalidator)",
    ),
    Cause(
        "history_truncated",
        "earlier messages were removed",
        "Dropping old turns changes the prefix from that point on. Compaction and context "
        "editing are the supported ways to shrink history. (shared/agent-design.md, Long-Running "
        "Agents)",
    ),
    Cause(
        "new_conversation",
        "the first message differs, which looks like a new conversation",
        "Expected at a conversation boundary, if tools and system still read from cache. Use "
        "--group-by when several conversations are interleaved in one file.",
        counts_as_waste=False,
    ),
    Cause(
        "model_switch",
        "the model changed",
        f"Caches are model-scoped. Keep one model per conversation. ({CITE}, Invalidation "
        "hierarchy)",
        counts_as_waste=False,
    ),
    Cause(
        "no_breakpoint",
        "the prefix is stable but no request marks a cache breakpoint",
        "Add a top-level `cache_control` or a marker at the end of the stable part. "
        f"({CITE}, Placement patterns)",
    ),
    Cause(
        "below_minimum",
        "the prefix is stable but shorter than this model's minimum cacheable size",
        f"Nothing to fix unless the prompt grows. ({CITE}, API reference)",
        counts_as_waste=False,
    ),
    Cause(
        "ttl_expired",
        "the prefix is stable but the gap exceeded the cache TTL",
        f"Use the 1-hour TTL for 5 to 60 minute gaps, or re-warm. ({CITE}, Choosing the TTL)",
    ),
    Cause(
        "params_changed",
        "the prefix is stable but `tool_choice`, `thinking` or `effort` changed",
        "Pin these per route; changing them invalidates the messages cache. "
        f"({CITE}, Invalidation hierarchy)",
    ),
    Cause(
        "lookback_exceeded",
        "the prefix is stable but the turn added more than 20 positions",
        "Place an intermediate breakpoint about every 15 positions in long turns. "
        f"({CITE}, 20-block lookback window)",
    ),
    Cause(
        "unexplained",
        "the prefix is stable and has a breakpoint, yet nothing was read",
        "Not visible in these bytes. Check for concurrent requests (an entry is readable only "
        "once the first response starts streaming), split workspaces, or a server-side "
        f"invalidation. ({CITE}, Concurrent-request timing)",
    ),
)
CAUSES: dict[str, Cause] = {cause.id: cause for cause in _CAUSES}

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
TIMESTAMP_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?",
        r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b",
        rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:,?\s+\d{{4}})?(?:\s+\d{{1,2}}:\d{{2}}(?::\d{{2}})?)?",
        r"\b\d{1,2}\s+(?:" + _MONTHS + r")\s+\d{4}\b",
        r"\b1[5-9]\d{8}(?:\d{3})?(?:\.\d+)?\b",  # epoch seconds or milliseconds
    )
)
IDENTIFIER_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        # A named ID whose value holds a digit, so that "run the report" does not count.
        r"\b(?:request|req|trace|span|session|sess|run|correlation|corr|msg)"
        r"(?:[-_ ]?id)?[-_:= \"']+(?=[A-Za-z_-]*\d)[A-Za-z0-9][A-Za-z0-9_-]{5,}",
        r"\b[0-9a-f]{16,}\b",
    )
)
VOLATILE_WINDOW = 96


@dataclass
class Record:
    """One line of the input."""

    line: int
    request: dict[str, Any]
    usage: dict[str, Any] | None
    started_at: float | None
    group: str = ""
    rendered: Rendered = field(init=False)

    def __post_init__(self) -> None:
        self.rendered = render(self.request)

    @property
    def model(self) -> str | None:
        value = self.request.get("model")
        return value if isinstance(value, str) else None

    def tokens(self, key: str) -> int:
        value = (self.usage or {}).get(key)
        return value if isinstance(value, int) else 0

    @property
    def input_tokens(self) -> int:
        return self.tokens("input_tokens")

    @property
    def written(self) -> int:
        return self.tokens("cache_creation_input_tokens")

    @property
    def read(self) -> int:
        return self.tokens("cache_read_input_tokens")

    @property
    def total(self) -> int:
        return self.input_tokens + self.written + self.read

    @property
    def write_multiplier(self) -> float:
        detail = (self.usage or {}).get("cache_creation")
        if isinstance(detail, dict) and self.written:
            one_hour = detail.get("ephemeral_1h_input_tokens") or 0
            share = min(one_hour / self.written, 1.0)
            return share * facts.WRITE_MULTIPLIER_1H + (1 - share) * facts.WRITE_MULTIPLIER_5M
        return facts.WRITE_MULTIPLIER_5M


@dataclass
class Break:
    previous: Record
    current: Record
    cause: Cause
    offset: int | None = None  # None when no byte differs
    path: str | None = None
    detail: str = ""
    before: str = ""
    after: str = ""
    caret: int = 0
    wasted_tokens: int = 0  # tokens billed above the read rate that could have been reads
    waste_units: float = 0.0  # the same, in base-input-token equivalents
    waste_usd: float | None = None

    def to_dict(self, redact: bool = False) -> dict[str, Any]:
        data = {
            "previous_line": self.previous.line,
            "line": self.current.line,
            "group": self.current.group,
            "cause": self.cause.id,
            "summary": self.cause.summary,
            "fix": self.cause.fix,
            "first_changed_byte": self.offset,
            "previous_bytes": len(self.previous.rendered.data),
            "bytes": len(self.current.rendered.data),
            "path": self.path,
            "detail": self.detail,
            "cache_read_input_tokens": self.current.read,
            "estimated_wasted_tokens": self.wasted_tokens,
            "estimated_waste_base_tokens": round(self.waste_units, 1),
            "estimated_waste_usd": None if self.waste_usd is None else round(self.waste_usd, 6),
        }
        if redact:
            del data["detail"]  # may quote the changed value itself
        else:
            data["before"] = self.before
            data["after"] = self.after
        return data


@dataclass
class Report:
    records: list[Record]
    breaks: list[Break]
    pairs: int

    @property
    def prompt_tokens(self) -> int:
        return sum(r.total for r in self.records)

    @property
    def read_tokens(self) -> int:
        return sum(r.read for r in self.records)

    @property
    def written_tokens(self) -> int:
        return sum(r.written for r in self.records)

    @property
    def uncached_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def hit_rate(self) -> float | None:
        return self.read_tokens / self.prompt_tokens if self.prompt_tokens else None

    @property
    def waste_units(self) -> float:
        return sum(b.waste_units for b in self.breaks)

    @property
    def waste_usd(self) -> float | None:
        known = [b.waste_usd for b in self.breaks if b.waste_usd is not None]
        return sum(known) if known else None

    def cause_counts(self) -> Counter[str]:
        return Counter(b.cause.id for b in self.breaks)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _covers(pattern: re.Pattern[str], text: str, position: int) -> str | None:
    for match in pattern.finditer(text):
        if match.start() <= position < match.end():
            return match.group(0)
    return None


def _window(data: bytes, offset: int) -> tuple[str, int]:
    start = max(0, offset - VOLATILE_WINDOW)
    while start > 0 and data[start] & 0xC0 == 0x80:
        start -= 1
    left = data[start:offset].decode("utf-8", errors="replace")
    right = data[offset : offset + VOLATILE_WINDOW].decode("utf-8", errors="replace")
    return left + right, len(left)


def _volatile(previous: bytes, current: bytes, offset: int) -> tuple[str, str] | None:
    """If the same kind of volatile value spans the first difference on both sides, name it."""
    left_text, left_position = _window(previous, offset)
    right_text, right_position = _window(current, offset)
    for cause, patterns in (("timestamp", TIMESTAMP_PATTERNS), ("request_id", IDENTIFIER_PATTERNS)):
        for pattern in patterns:
            left = _covers(pattern, left_text, left_position)
            right = _covers(pattern, right_text, right_position)
            if left and right and left != right:
                return cause, f"{left} -> {right}"
    return None


def _tool_name(tool: Any) -> str:
    return str(tool.get("name", "?")) if isinstance(tool, dict) else "?"


def _classify_tools(previous: Record, current: Record) -> tuple[str, str] | None:
    before = previous.request.get("tools") or []
    after = current.request.get("tools") or []
    if [_canonical(t) for t in before] == [_canonical(t) for t in after]:
        return None
    if Counter(map(_canonical, before)) == Counter(map(_canonical, after)):
        return "unsorted_tools", (
            f"{[_tool_name(t) for t in before]} -> {[_tool_name(t) for t in after]}"
        )
    names_before = Counter(map(_tool_name, before))
    names_after = Counter(map(_tool_name, after))
    if names_before != names_after:
        added = sorted((names_after - names_before).elements())
        removed = sorted((names_before - names_after).elements())
        return "tool_set_changed", f"added {added}, removed {removed}"
    if sorted(names_before) == sorted(names_after) and [_tool_name(t) for t in before] != [
        _tool_name(t) for t in after
    ]:
        return "unsorted_tools", "order changed, and at least one definition changed too"
    return None


def _classify_bytes(previous: Record, current: Record, offset: int) -> tuple[str, str]:
    prev = previous.rendered
    curr = current.rendered
    prev_atom = prev.atom_at(offset)
    curr_atom = curr.atom_at(offset)
    root = (curr_atom or prev_atom).path if (curr_atom or prev_atom) else ""

    tools = _classify_tools(previous, current)
    if tools and tools[0] == "unsorted_tools":
        return tools

    if prev_atom and curr_atom and prev_atom.path == curr_atom.path:
        if prev_atom.value != curr_atom.value:
            volatile = _volatile(prev.data, curr.data, offset)
            if volatile:
                return volatile
        elif _canonical(prev_atom.value) == _canonical(curr_atom.value):
            return "json_key_order", f"in {prev_atom.path}"

    if tools:
        return tools
    if root.startswith("tools"):
        return "tool_edited", _inserted_hint(curr.data, offset)
    if root.startswith("system"):
        return "system_edited", _inserted_hint(curr.data, offset)

    before_count = len(previous.request.get("messages") or [])
    after_count = len(current.request.get("messages") or [])
    if after_count < before_count:
        return "history_truncated", f"{before_count} messages -> {after_count}"
    if root == "messages[0]" or root.startswith("messages[0]."):
        return "new_conversation", ""
    return "history_rewritten", _inserted_hint(curr.data, offset)


def _inserted_hint(data: bytes, offset: int) -> str:
    """A value that only one side has can't be matched against the other. Still worth saying."""
    text = data[offset : offset + VOLATILE_WINDOW].decode("utf-8", errors="replace")
    for label, patterns in (("timestamp", TIMESTAMP_PATTERNS), ("ID", IDENTIFIER_PATTERNS)):
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return f"the new text contains what looks like a {label}: {match.group(0)}"
    return ""


def _parameter(request: dict[str, Any], key: str) -> Any:
    if key == "effort":
        config = request.get("output_config")
        return config.get("effort") if isinstance(config, dict) else None
    return request.get(key)


def _classify_clean(previous: Record, current: Record) -> tuple[str, str] | None:
    """The bytes extend cleanly. Return a cause only if the usage shows something went wrong."""
    if current.usage is None or previous.usage is None or previous.total == 0:
        return None
    changed = [
        key
        for key in ("tool_choice", "thinking", "effort")
        if _parameter(previous.request, key) != _parameter(current.request, key)
    ]
    if changed and current.read < previous.total:
        return "params_changed", f"changed: {', '.join(changed)}"
    if current.read > 0:
        return None
    if not previous.rendered.has_breakpoint:
        return "no_breakpoint", ""
    minimum = facts.MINIMUM_CACHEABLE.get(previous.model or "")
    if minimum and previous.total < minimum:
        return "below_minimum", f"{previous.total} tokens, minimum {minimum}"
    if previous.started_at is not None and current.started_at is not None:
        gap = current.started_at - previous.started_at
        ttl = facts.TTL_1H_SECONDS if previous.rendered.one_hour else facts.TTL_5M_SECONDS
        if gap > ttl:
            return "ttl_expired", f"{gap:.0f}s between starts, TTL {ttl}s"
    added = len(current.rendered.atoms) - len(previous.rendered.atoms)
    if added > facts.LOOKBACK_POSITIONS:
        return "lookback_exceeded", f"{added} new positions at most"
    return "unexplained", ""


def _snippets(previous: bytes, current: bytes, offset: int) -> tuple[str, str, int]:
    """Short escaped excerpts around the first difference, and where it falls in them."""

    def piece(data: bytes, start: int, end: int) -> str:
        return data[start:end].decode("utf-8", errors="replace").replace("\n", "\\n")

    start = max(0, offset - 36)
    while start > 0 and previous[start] & 0xC0 == 0x80:
        start -= 1
    left = piece(previous, start, offset)
    return (
        left + piece(previous, offset, offset + 28),
        left + piece(current, offset, offset + 28),
        len(left),
    )


def estimate_waste(item: Break, input_price: float | None) -> None:
    """How much cheaper this request would have been if the previous prompt had been read.

    Caps at what the request actually paid above the read rate, and subtracts what it did read.
    Writes are spent first because they are dearer. It ignores that a fix may change breakpoints,
    and the premium paid to write the entry in the first place, so read it as a rough size.
    """
    if not item.cause.counts_as_waste or item.current.usage is None:
        return
    current = item.current
    rereadable = max(0, item.previous.total - current.read)
    tokens = min(rereadable, current.written + current.input_tokens)
    if tokens <= 0:
        return
    from_writes = min(tokens, current.written)
    from_input = tokens - from_writes
    read = facts.READ_MULTIPLIER.get(current.model or "", facts.READ_MULTIPLIER_DEFAULT)
    item.wasted_tokens = tokens
    item.waste_units = from_writes * (current.write_multiplier - read) + from_input * (1 - read)
    price = input_price if input_price is not None else facts.INPUT_PRICE.get(current.model or "")
    if price is not None:
        item.waste_usd = item.waste_units * price / 1_000_000


def compare(previous: Record, current: Record, input_price: float | None = None) -> Break | None:
    item: Break | None = None
    if previous.model != current.model:
        item = Break(
            previous, current, CAUSES["model_switch"], detail=f"{previous.model} -> {current.model}"
        )
    else:
        old, new = previous.rendered.data, current.rendered.data
        offset = common_prefix_length(old, new)
        if offset == len(old):
            clean = _classify_clean(previous, current)
            if clean:
                item = Break(previous, current, CAUSES[clean[0]], detail=clean[1])
        else:
            while offset > 0 and old[offset] & 0xC0 == 0x80:
                offset -= 1
            cause, detail = _classify_bytes(previous, current, offset)
            before, after, caret = _snippets(old, new, offset)
            path = current.rendered.path_at(offset) if offset < len(new) else "<end of request>"
            item = Break(
                previous, current, CAUSES[cause], offset, path, detail, before, after, caret
            )
    if item:
        estimate_waste(item, input_price)
    return item


def analyze(records: list[Record], input_price: float | None = None) -> Report:
    groups: dict[str, list[Record]] = {}
    for record in records:
        groups.setdefault(record.group, []).append(record)
    breaks: list[Break] = []
    pairs = 0
    for members in groups.values():
        for previous, current in pairwise(members):
            pairs += 1
            item = compare(previous, current, input_price)
            if item:
                breaks.append(item)
    breaks.sort(key=lambda b: b.current.line)
    return Report(records, breaks, pairs)


def parse_time(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.timestamp()
    return None


def _dig(record: dict[str, Any], dotted: str) -> Any:
    current: Any = record
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


class InputError(Exception):
    """A line could not be understood."""


def parse_line(raw: str, number: int, group_by: str | None = None) -> Record:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(f"line {number}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError(f"line {number}: expected a JSON object")
    request = data.get("request", data)
    if not isinstance(request, dict) or "messages" not in request:
        raise InputError(f"line {number}: no request with `messages` found")
    usage = data.get("usage")
    response = data.get("response")
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    started = next(
        (data[key] for key in ("timestamp", "ts", "started_at") if key in data),
        None,
    )
    group = ""
    if group_by:
        value = _dig(data, group_by)
        if value is None:
            value = _dig(request, group_by)
        group = "" if value is None else str(value)
    return Record(
        line=number,
        request=request,
        usage=usage if isinstance(usage, dict) else None,
        started_at=parse_time(started),
        group=group,
    )


def load_records(text: str, group_by: str | None = None) -> list[Record]:
    return [
        parse_line(raw, number, group_by)
        for number, raw in enumerate(text.splitlines(), start=1)
        if raw.strip()
    ]
