from __future__ import annotations

from typing import Any

import pytest

from prompt_cache_lens.analyze import InputError, Record, analyze, compare, load_records, parse_time

FILLER = "Stable instructions for the synthetic desk. " * 5
USER = {"role": "user", "content": "Synthetic question."}


def request(system: Any = FILLER, tools: list | None = None, messages: list | None = None, **rest):
    return {
        "model": "claude-opus-5-5",
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if isinstance(system, str)
        else system,
        "tools": tools or [],
        "messages": messages if messages is not None else [USER],
        **rest,
    }


def record(req: dict, line: int = 1, usage: dict | None = None, at: float | None = None) -> Record:
    return Record(line=line, request=req, usage=usage, started_at=at)


def cause_of(before: dict, after: dict, **kwargs) -> str | None:
    item = compare(record(before), record(after, 2), **kwargs)
    return item.cause.id if item else None


TOOL_A = {"name": "a_tool", "description": "A.", "input_schema": {"type": "object"}}
TOOL_B = {"name": "b_tool", "description": "B.", "input_schema": {"type": "object"}}


def usage(read: int = 0, written: int = 0, uncached: int = 0, **extra) -> dict:
    return {
        "input_tokens": uncached,
        "cache_creation_input_tokens": written,
        "cache_read_input_tokens": read,
        **extra,
    }


def test_identical_requests_do_not_break():
    assert cause_of(request(), request()) is None


def test_appended_turns_do_not_break():
    longer = [USER, {"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
    assert cause_of(request(), request(messages=longer)) is None


def test_moving_cache_control_marker_does_not_break():
    marked = [{"role": "user", "content": [{"type": "text", "text": "q", "cache_control": {}}]}]
    plain = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
    assert cause_of(request(messages=marked), request(messages=plain)) is None


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("2026-10-30T09:00:24Z", "2026-10-30T09:00:51Z"),
        ("2026-10-30T09:00:24.123+00:00", "2026-10-30T09:14:24.456+00:00"),
        ("2026-10-30", "2026-10-31"),
        ("October 30, 2026", "October 31, 2026"),
        ("30 October 2026", "31 October 2026"),
        ("09:14:02", "09:14:37"),
        ("1793350800", "1793350861"),
        ("1793350800123", "1793350861456"),
    ],
)
def test_timestamp_formats(left, right):
    before = request(f"Now: {left}.\n{FILLER}")
    after = request(f"Now: {right}.\n{FILLER}")
    item = compare(record(before), record(after, 2))
    assert item.cause.id == "timestamp"
    assert item.path == "system[0].text"
    assert left in item.detail
    assert right in item.detail


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("8f6e2a87-c60d-5e53-357a-dacabc9ae844", "5b062361-9d83-609a-439e-f482cee95ac0"),
        ("request_id=abc12345", "request_id=abd99999"),
        ("req_01ABCDEF", "req_01ZZZZZZ"),
        ("trace: 7f3a91c2", "trace: 7f3b0000"),
        ("0123456789abcdef0123", "0123456789abffff0123"),
    ],
)
def test_identifier_formats(left, right):
    item = compare(record(request(f"{left}\n{FILLER}")), record(request(f"{right}\n{FILLER}"), 2))
    assert item.cause.id == "request_id", item.detail


def test_timestamp_in_a_user_message_is_named():
    before = request(messages=[{"role": "user", "content": "Sent at 2026-10-30T09:00:00Z. Hi"}])
    after = request(messages=[{"role": "user", "content": "Sent at 2026-10-30T09:09:00Z. Hi"}])
    assert cause_of(before, after) == "timestamp"


def test_ordinary_words_are_not_identifiers():
    before = request(f"Please run the report now.\n{FILLER}")
    after = request(f"Please run the summary now.\n{FILLER}")
    assert cause_of(before, after) == "system_edited"


def test_unsorted_tools():
    item = compare(record(request(tools=[TOOL_A, TOOL_B])), record(request(tools=[TOOL_B, TOOL_A])))
    assert item.cause.id == "unsorted_tools"
    assert "a_tool" in item.detail
    assert item.path.startswith("tools[0]")


def test_key_order_in_a_tool():
    reordered = {"input_schema": {"type": "object"}, "description": "A.", "name": "a_tool"}
    assert cause_of(request(tools=[TOOL_A]), request(tools=[reordered])) == "json_key_order"


def test_key_order_nested_in_a_tool_use_input():
    def turn(input_value: dict) -> list:
        return [
            USER,
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": input_value}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]},
        ]

    before = request(messages=turn({"a": 1, "b": 2}))
    after = request(messages=turn({"b": 2, "a": 1}))
    item = compare(record(before), record(after, 2))
    assert item.cause.id == "json_key_order"
    assert item.path.startswith("messages[1].content[0].input")


def test_tool_added_and_removed():
    item = compare(record(request(tools=[TOOL_A])), record(request(tools=[TOOL_A, TOOL_B])))
    assert item.cause.id == "tool_set_changed"
    assert "b_tool" in item.detail


def test_tool_edited():
    edited = {**TOOL_A, "description": "A changed."}
    assert cause_of(request(tools=[TOOL_A]), request(tools=[edited])) == "tool_edited"


def test_system_edit_hints_at_a_newly_inserted_date():
    item = compare(record(request()), record(request(f"Today is 2026-10-30.\n{FILLER}"), 2))
    assert item.cause.id == "system_edited"
    assert "2026-10-30" in item.detail


def test_history_rewritten_and_truncated():
    original = [USER, {"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
    edited = [USER, {"role": "assistant", "content": "CHANGED"}, {"role": "user", "content": "y"}]
    assert cause_of(request(messages=original), request(messages=edited)) == "history_rewritten"
    assert cause_of(request(messages=original), request(messages=[USER])) == "history_truncated"


def test_new_conversation_is_not_waste():
    before = record(request(), usage=usage(written=2000))
    other = [{"role": "user", "content": "A different opening question."}]
    after = record(request(messages=other), 2, usage=usage(read=500, written=1500))
    item = compare(before, after)
    assert item.cause.id == "new_conversation"
    assert item.wasted_tokens == 0


def test_model_switch():
    item = compare(record(request()), record(request(model="claude-sonnet-5-5"), 2))
    assert item.cause.id == "model_switch"
    assert item.offset is None


def test_offset_and_snippets_point_at_the_first_changed_byte():
    item = compare(record(request("abcdefXghij")), record(request("abcdefYghij"), 2))
    assert item.previous.rendered.data[item.offset : item.offset + 1] == b"X"
    assert item.current.rendered.data[item.offset : item.offset + 1] == b"Y"
    assert item.before[item.caret] == "X"
    assert item.after[item.caret] == "Y"


def test_offset_does_not_split_a_multibyte_character():
    item = compare(record(request("café one")), record(request("cafè one"), 2))
    assert item.before[item.caret] == "é"
    assert item.after[item.caret] == "è"


def test_clean_prefix_without_markers_is_no_breakpoint():
    plain = {"model": "claude-opus-5-5", "system": FILLER, "messages": [USER]}
    item = compare(
        record(plain, usage=usage(uncached=2000)), record(plain, 2, usage=usage(uncached=2000))
    )
    assert item.cause.id == "no_breakpoint"


def test_clean_prefix_below_minimum():
    item = compare(
        record(request(), usage=usage(uncached=100)),
        record(request(), 2, usage=usage(uncached=100)),
    )
    assert item.cause.id == "below_minimum"
    assert item.waste_units == 0


def test_clean_prefix_ttl_expired_and_one_hour_ttl():
    item = compare(
        record(request(), usage=usage(written=2000), at=0.0),
        record(request(), 2, usage=usage(written=2000), at=400.0),
    )
    assert item.cause.id == "ttl_expired"

    block = {"type": "text", "text": FILLER, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    one_hour = request(system=[block])
    item = compare(
        record(one_hour, usage=usage(written=2000), at=0.0),
        record(one_hour, 2, usage=usage(written=2000), at=400.0),
    )
    assert item.cause.id == "unexplained"


def test_clean_prefix_params_changed():
    before = request(tool_choice={"type": "auto"})
    after = request(tool_choice={"type": "none"})
    item = compare(
        record(before, usage=usage(written=2000)), record(after, 2, usage=usage(read=500))
    )
    assert item.cause.id == "params_changed"
    assert "tool_choice" in item.detail


def test_clean_prefix_with_effort_changed():
    before = request(output_config={"effort": "low"})
    after = request(output_config={"effort": "high"})
    item = compare(record(before, usage=usage(written=2000)), record(after, 2, usage=usage()))
    assert item.cause.id == "params_changed"


def test_clean_prefix_lookback_exceeded():
    many = [USER, *({"role": "user", "content": f"m{i}"} for i in range(25))]
    item = compare(
        record(request(), usage=usage(written=2000)),
        record(request(messages=many), 2, usage=usage(written=3000)),
    )
    assert item.cause.id == "lookback_exceeded"


def test_clean_prefix_with_a_hit_is_healthy():
    assert (
        compare(
            record(request(), usage=usage(written=2000)),
            record(request(), 2, usage=usage(read=2000)),
        )
        is None
    )


def test_clean_prefix_without_usage_says_nothing():
    assert compare(record(request()), record(request(), 2)) is None


def test_waste_for_writes_and_uncached_tokens():
    before = record(request("one 2026-10-30"), usage=usage(written=1000))
    after = record(request("one 2026-10-31"), 2, usage=usage(written=800, uncached=500))
    item = compare(before, after)
    assert item.wasted_tokens == 1000  # capped by what the previous request could have supplied
    # 800 written at 1.25 and 200 uncached at 1.0, each instead of a 0.05 read on this model.
    assert item.waste_units == pytest.approx(800 * 1.20 + 200 * 0.95)
    assert item.waste_usd == pytest.approx(item.waste_units * 4.0 / 1_000_000)


def test_waste_subtracts_what_was_read():
    before = record(request("one 2026-10-30"), usage=usage(written=1000))
    after = record(request("one 2026-10-31"), 2, usage=usage(read=700, written=500))
    assert compare(before, after).wasted_tokens == 300


def test_waste_uses_the_one_hour_write_price():
    detail = {"ephemeral_1h_input_tokens": 500, "ephemeral_5m_input_tokens": 0}
    before = record(request("one 2026-10-30"), usage=usage(written=500))
    after = record(request("one 2026-10-31"), 2, usage=usage(written=500, cache_creation=detail))
    assert compare(before, after).waste_units == pytest.approx(500 * (2.0 - 0.05))


def test_unpriced_model_gives_no_dollars_unless_a_price_is_passed():
    old = {**request("one 2026-10-30"), "model": "claude-opus-4-8"}
    new = {**request("one 2026-10-31"), "model": "claude-opus-4-8"}
    pair = (record(old, usage=usage(written=1000)), record(new, 2, usage=usage(written=1000)))
    assert compare(*pair).waste_usd is None
    assert compare(*pair).waste_units == pytest.approx(1000 * (1.25 - 0.1))
    assert compare(*pair, input_price=2.0).waste_usd == pytest.approx(1000 * 1.15 * 2 / 1e6)


def test_report_hit_rate_and_causes():
    records = [
        record(request(), 1, usage(written=1000)),
        record(request(), 2, usage(read=1000, uncached=100)),
        record(request("edited"), 3, usage(written=900)),
    ]
    report = analyze(records)
    assert report.hit_rate == pytest.approx(1000 / 3000)
    assert report.pairs == 2
    assert dict(report.cause_counts()) == {"system_edited": 1}


def test_report_groups_interleaved_conversations():
    def line(number: int, name: str, text: str) -> str:
        import json

        message = {"role": "user", "content": text}
        body = {"model": "m", "system": name, "messages": [message]}
        return json.dumps({"conversation": name, "request": body, "line_marker": number})

    lines = [line(1, "a", "q"), line(2, "b", "q"), line(3, "a", "q"), line(4, "b", "q")]
    text = "\n".join(lines)
    assert analyze(load_records(text)).breaks
    grouped = analyze(load_records(text, group_by="conversation"))
    assert grouped.pairs == 2
    assert grouped.breaks == []


def test_no_usage_means_no_hit_rate():
    assert analyze([record(request())]).hit_rate is None


def test_parse_line_shapes():
    flat = '{"model": "m", "messages": [], "usage": {"input_tokens": 3}}'
    nested = '{"request": {"messages": []}, "response": {"usage": {"input_tokens": 4}}}'
    records = load_records(flat + "\n\n" + nested)
    assert [r.input_tokens for r in records] == [3, 4]
    assert [r.line for r in records] == [1, 3]


@pytest.mark.parametrize("bad", ["{nope", "[1, 2]", '{"no": "messages"}'])
def test_parse_line_errors(bad):
    with pytest.raises(InputError, match="line 1"):
        load_records(bad)


def test_parse_time():
    assert parse_time(12.5) == 12.5
    assert parse_time("1970-01-01T00:01:00Z") == 60.0
    assert parse_time("not a time") is None
    assert parse_time(None) is None
