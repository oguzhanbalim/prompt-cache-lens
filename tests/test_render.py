from __future__ import annotations

from prompt_cache_lens.render import common_prefix_length, render, strip_cache_control


def test_render_order_is_tools_then_system_then_messages():
    request = {
        "messages": [{"role": "user", "content": "MESSAGE"}],
        "system": "SYSTEM",
        "tools": [{"name": "TOOL"}],
    }
    data = render(request).data
    assert data.index(b"TOOL") < data.index(b"SYSTEM") < data.index(b"MESSAGE")


def test_appended_turn_extends_the_previous_bytes():
    first = {"system": "s", "messages": [{"role": "user", "content": "a"}]}
    second = {
        "system": "s",
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ],
    }
    assert render(second).data.startswith(render(first).data)


def test_cache_control_is_not_rendered_but_is_recorded():
    marked = {
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "messages": [],
    }
    plain = {"system": [{"type": "text", "text": "s"}], "messages": []}
    assert render(marked).data == render(plain).data
    assert render(marked).breakpoints
    assert not render(plain).breakpoints


def test_ttl_and_automatic_caching_are_recorded():
    block = {"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    assert render({"system": [block], "messages": []}).one_hour
    automatic = render({"messages": [], "cache_control": {"type": "ephemeral"}})
    assert automatic.automatic
    assert automatic.has_breakpoint


def test_string_and_single_text_block_render_the_same():
    as_string = {"system": "s", "messages": [{"role": "user", "content": "hi"}]}
    as_block = {
        "system": [{"type": "text", "text": "s"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    assert render(as_string).data == render(as_block).data


def test_key_order_is_preserved_in_the_bytes():
    left = render({"tools": [{"a": 1, "b": 2}], "messages": []}).data
    right = render({"tools": [{"b": 2, "a": 1}], "messages": []}).data
    assert left != right


def test_path_at_points_at_keys_and_values():
    rendered = render(
        {"tools": [{"name": "x", "input_schema": {"type": "object"}}], "messages": []}
    )
    at_value = rendered.data.index(b'"object"')
    assert rendered.path_at(at_value) == "tools[0].input_schema.type"
    assert rendered.path_at(at_value - 1) == "tools[0].input_schema.type"


def test_strip_cache_control_recurses():
    nested = {"a": [{"cache_control": 1, "b": {"cache_control": 2, "c": 3}}]}
    assert strip_cache_control(nested) == {"a": [{"b": {"c": 3}}]}


def test_common_prefix_length():
    assert common_prefix_length(b"", b"") == 0
    assert common_prefix_length(b"abc", b"abc") == 3
    assert common_prefix_length(b"abcd", b"abxd") == 2
    assert common_prefix_length(b"abc", b"abcdef") == 3
    assert common_prefix_length(b"xbc", b"abc") == 0
