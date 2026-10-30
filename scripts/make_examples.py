#!/usr/bin/env python3
"""Generate the synthetic JSONL files in examples/.

Everything here is invented. Prompts are filler, identifiers come from a seeded generator, and
the `usage` blocks are computed by a toy prefix cache below (tokens are bytes / 4), not returned
by any API. They exist to show the tool's output. Run from the repository root:

    python scripts/make_examples.py
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prompt_cache_lens.render import render

MODEL = "claude-opus-5-5"
START = 1_793_350_800  # a fixed instant, so that output never depends on the clock
CLAUSES = " ".join(
    f"Clause {i}: the example desk answers synthetic question type {i} from the handbook."
    for i in range(1, 30)
)
SYSTEM_TEXT = "You are the Example Desk assistant, for a made-up help desk. " + CLAUSES


def iso(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def tool(name: str, field: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Synthetic tool {name}. Use when the made-up desk needs {field}.",
        "input_schema": {
            "type": "object",
            "properties": {field: {"type": "string", "description": f"The {field} to look up."}},
            "required": [field],
        },
    }


TOOLS = [
    tool("get_ticket", "ticket_ref"),
    tool("list_articles", "topic"),
    tool("send_note", "text"),
]


def conversation(turn: int) -> list[dict[str, Any]]:
    """An append-only history that is `turn` user messages long."""
    messages: list[dict[str, Any]] = []
    for index in range(turn):
        messages.append({"role": "user", "content": f"Synthetic question number {index + 1}."})
        if index < turn - 1:
            messages.append({"role": "assistant", "content": f"Synthetic answer {index + 1}."})
    return messages


def base_request(turn: int, system: str = SYSTEM_TEXT, tools: list[Any] | None = None) -> dict:
    messages = deepcopy(conversation(turn))
    messages[-1]["content"] = [
        {
            "type": "text",
            "text": messages[-1]["content"],
            "cache_control": {"type": "ephemeral"},
        }
    ]
    return {
        "model": MODEL,
        "max_tokens": 1024,
        "tools": deepcopy(TOOLS if tools is None else tools),
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": messages,
    }


class ToyCache:
    """Reads the longest stored breakpoint that prefixes this request, writes up to the last."""

    def __init__(self) -> None:
        self.entries: set[bytes] = set()

    def usage(self, request: dict[str, Any]) -> dict[str, int]:
        rendered = render(request)
        data = rendered.data
        read_bytes = max(
            (len(entry) for entry in self.entries if data.startswith(entry)), default=0
        )
        last = rendered.breakpoints[-1] if rendered.breakpoints else 0
        write_bytes = max(0, last - read_bytes)
        for point in rendered.breakpoints:
            self.entries.add(data[:point])
        tail_bytes = len(data) - max(last, read_bytes)
        return {
            "input_tokens": tail_bytes // 4,
            "cache_creation_input_tokens": write_bytes // 4,
            "cache_read_input_tokens": read_bytes // 4,
            "output_tokens": 40,
        }


def healthy(rng: random.Random) -> list[dict]:
    return [base_request(turn) for turn in range(1, 7)]


def timestamp_in_system(rng: random.Random) -> list[dict]:
    requests = []
    now = START
    for turn in range(1, 7):
        now += rng.randint(20, 90)
        requests.append(base_request(turn, f"Current time: {iso(now)}\n{SYSTEM_TEXT}"))
    return requests


def request_id_in_system(rng: random.Random) -> list[dict]:
    requests = []
    for turn in range(1, 7):
        identifier = "-".join(
            "".join(rng.choices("0123456789abcdef", k=size)) for size in (8, 4, 4, 4, 12)
        )
        requests.append(base_request(turn, f"Request ID: {identifier}\n{SYSTEM_TEXT}"))
    return requests


def unsorted_tools(rng: random.Random) -> list[dict]:
    requests = []
    order = list(TOOLS)
    for turn in range(1, 7):
        previous = list(order)
        while order == previous:
            rng.shuffle(order)
        requests.append(base_request(turn, tools=order))
    return requests


def varying_key_order(rng: random.Random) -> list[dict]:
    requests = []
    for turn in range(1, 7):
        tools = deepcopy(TOOLS)
        schema = tools[1]["input_schema"]
        if turn % 2 == 0:
            tools[1]["input_schema"] = {key: schema[key] for key in reversed(list(schema))}
        requests.append(base_request(turn, tools=tools))
    return requests


def regression_mid_session(rng: random.Random) -> list[dict]:
    """Healthy for three turns, then someone adds the date to the system prompt."""
    requests = []
    for turn in range(1, 7):
        system = SYSTEM_TEXT
        if turn > 3:
            system = f"Today is {iso(START + turn * 86400)[:10]}.\n{SYSTEM_TEXT}"
        requests.append(base_request(turn, system))
    return requests


SCENARIOS: dict[str, Callable[[random.Random], list[dict]]] = {
    "healthy": healthy,
    "timestamp_in_system": timestamp_in_system,
    "request_id_in_system": request_id_in_system,
    "unsorted_tools": unsorted_tools,
    "varying_key_order": varying_key_order,
    "regression_mid_session": regression_mid_session,
}


def build(name: str) -> str:
    rng = random.Random(f"prompt-cache-lens/{name}")
    cache = ToyCache()
    lines = []
    for index, request in enumerate(SCENARIOS[name](rng)):
        record = {
            "timestamp": iso(START + index * 60),
            "synthetic": True,
            "request": request,
            "usage": cache.usage(request),
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples")
    output.mkdir(parents=True, exist_ok=True)
    for name in SCENARIOS:
        (output / f"{name}.jsonl").write_text(build(name), encoding="utf-8")
    print(f"wrote {len(SCENARIOS)} files to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
