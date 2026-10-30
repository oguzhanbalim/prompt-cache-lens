"""Lay a request out as bytes in render order, remembering what sits at each offset.

The API's real rendering happens server side and is not visible here. This is a stand-in with
the property that matters: order and content follow the request, tools then system then
messages, and an append-only conversation produces a byte-for-byte extension of the previous
request. Offsets are offsets into this rendering, not into anything the API computes.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Atom:
    """One unit that gets compared: a tool, a system block, a role marker or a content block."""

    path: str
    start: int
    end: int
    value: Any


@dataclass
class Rendered:
    data: bytes
    atoms: list[Atom]
    spans: list[tuple[int, int, str]]  # (start, end, path) for every key and scalar, sorted
    breakpoints: list[int]  # offsets where a cache_control marker ends a block
    one_hour: bool = False  # any breakpoint asks for the 1h TTL
    automatic: bool = False  # top-level cache_control

    def path_at(self, offset: int) -> str:
        """The most specific path at this offset; structure characters take the next path."""
        if not self.spans:
            return "<empty>"
        starts = [span[0] for span in self.spans]
        index = bisect.bisect_right(starts, offset) - 1
        if index >= 0 and self.spans[index][0] <= offset < self.spans[index][1]:
            return self.spans[index][2]
        if index + 1 < len(self.spans):
            return self.spans[index + 1][2]
        return self.spans[-1][2]

    def atom_at(self, offset: int) -> Atom | None:
        starts = [atom.start for atom in self.atoms]
        index = bisect.bisect_right(starts, offset) - 1
        if index >= 0 and offset <= self.atoms[index].end:
            return self.atoms[index]
        return None

    def atom_named(self, path: str) -> Atom | None:
        return next((atom for atom in self.atoms if atom.path == path), None)

    @property
    def has_breakpoint(self) -> bool:
        return self.automatic or bool(self.breakpoints)


@dataclass
class _Writer:
    buffer: bytearray = field(default_factory=bytearray)
    spans: list[tuple[int, int, str]] = field(default_factory=list)

    def raw(self, text: str) -> None:
        self.buffer += text.encode("utf-8")

    def scalar(self, text: str, path: str) -> None:
        start = len(self.buffer)
        self.raw(text)
        self.spans.append((start, len(self.buffer), path))

    def value(self, item: Any, path: str) -> None:
        if isinstance(item, dict):
            self.raw("{")
            for index, (key, child) in enumerate(item.items()):
                if index:
                    self.raw(",")
                child_path = f"{path}.{key}"
                self.scalar(json.dumps(key, ensure_ascii=False), child_path)
                self.raw(":")
                self.value(child, child_path)
            self.raw("}")
        elif isinstance(item, list):
            self.raw("[")
            for index, child in enumerate(item):
                if index:
                    self.raw(",")
                self.value(child, f"{path}[{index}]")
            self.raw("]")
        else:
            self.scalar(json.dumps(item, ensure_ascii=False), path)


def strip_cache_control(item: Any) -> Any:
    """Markers move between requests without invalidating anything, so they never get compared."""
    if isinstance(item, dict):
        return {k: strip_cache_control(v) for k, v in item.items() if k != "cache_control"}
    if isinstance(item, list):
        return [strip_cache_control(v) for v in item]
    return item


def _blocks(content: Any) -> list[Any]:
    """A bare string and a one-element text block are written the same way here."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return [content]


def render(request: dict[str, Any]) -> Rendered:
    writer = _Writer()
    atoms: list[Atom] = []
    breakpoints: list[int] = []
    one_hour = False

    def atom(item: Any, path: str) -> None:
        nonlocal one_hour
        start = len(writer.buffer)
        clean = strip_cache_control(item)
        writer.value(clean, path)
        writer.raw("\n")
        atoms.append(Atom(path, start, len(writer.buffer), clean))
        if isinstance(item, dict) and "cache_control" in item:
            breakpoints.append(len(writer.buffer))
            control = item["cache_control"]
            if isinstance(control, dict) and control.get("ttl") == "1h":
                one_hour = True

    writer.raw("tools:\n")
    for index, tool in enumerate(request.get("tools") or []):
        atom(tool, f"tools[{index}]")
    writer.raw("system:\n")
    for index, block in enumerate(_blocks(request.get("system"))):
        atom(block, f"system[{index}]")
    writer.raw("messages:\n")
    for index, message in enumerate(request.get("messages") or []):
        role = message.get("role") if isinstance(message, dict) else None
        atom({"role": role}, f"messages[{index}]")
        content = message.get("content") if isinstance(message, dict) else message
        for block_index, block in enumerate(_blocks(content)):
            atom(block, f"messages[{index}].content[{block_index}]")

    top = request.get("cache_control")
    if isinstance(top, dict) and top.get("ttl") == "1h":
        one_hour = True
    return Rendered(
        data=bytes(writer.buffer),
        atoms=atoms,
        spans=sorted(writer.spans),
        breakpoints=breakpoints,
        one_hour=one_hour,
        automatic=top is not None,
    )


def common_prefix_length(left: bytes, right: bytes) -> int:
    """Longest common prefix, by binary search over slice comparisons."""
    low, high = 0, min(len(left), len(right))
    while low < high:
        middle = (low + high + 1) // 2
        if left[low:middle] == right[low:middle]:
            low = middle
        else:
            high = middle - 1
    return low
