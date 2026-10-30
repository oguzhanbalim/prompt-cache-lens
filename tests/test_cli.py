from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from prompt_cache_lens.cli import EXIT_BELOW_THRESHOLD, EXIT_OK, EXIT_USAGE, main

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_examples", ROOT / "scripts/make_examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_report(capsys, *args):
    main([*args, "--format", "json"])
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("name", "cause"),
    [
        ("timestamp_in_system", "timestamp"),
        ("request_id_in_system", "request_id"),
        ("unsorted_tools", "unsorted_tools"),
        ("varying_key_order", "json_key_order"),
    ],
)
def test_each_example_names_its_cause(capsys, name, cause):
    report = json_report(capsys, str(EXAMPLES / f"{name}.jsonl"))
    assert report["causes"] == {cause: 5}


def test_healthy_example_has_no_breaks(capsys):
    report = json_report(capsys, str(EXAMPLES / "healthy.jsonl"))
    assert report["causes"] == {}
    assert report["hit_rate"] > 0.5
    assert report["estimated_waste_base_tokens"] == 0


def test_regression_example_finds_the_turn_it_started(capsys):
    report = json_report(capsys, str(EXAMPLES / "regression_mid_session.jsonl"))
    first = report["breaks"][0]
    assert first["line"] == 4
    assert first["cause"] == "system_edited"
    assert "2026" in first["detail"]
    assert report["causes"]["timestamp"] == 2


def test_examples_are_marked_synthetic():
    for path in EXAMPLES.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            assert json.loads(line)["synthetic"] is True


def test_examples_are_exactly_what_the_generator_produces():
    generator = load_generator()
    for name in generator.SCENARIOS:
        assert (EXAMPLES / f"{name}.jsonl").read_text() == generator.build(name), name


def test_generator_cache_rewards_a_stable_prefix():
    generator = load_generator()
    cache = generator.ToyCache()
    first = cache.usage(generator.base_request(1))
    second = cache.usage(generator.base_request(2))
    assert first["cache_read_input_tokens"] == 0
    assert first["cache_creation_input_tokens"] > 0
    assert second["cache_read_input_tokens"] == (
        first["cache_creation_input_tokens"] + first["input_tokens"]
    )


def test_text_output_shows_the_byte_and_a_caret(capsys):
    assert main([str(EXAMPLES / "timestamp_in_system.jsonl")]) == EXIT_OK
    out = capsys.readouterr().out
    assert "first changed byte:" in out
    assert "at system[0].text" in out
    assert "^" in out
    assert "the same cause in the same section on 4 more pair(s)" in out


def test_all_shows_every_pair(capsys):
    main([str(EXAMPLES / "timestamp_in_system.jsonl"), "--all"])
    assert capsys.readouterr().out.count("first changed byte:") == 5


def test_redact_removes_prompt_text(capsys):
    path = str(EXAMPLES / "timestamp_in_system.jsonl")
    main([path, "--redact"])
    assert "Current time" not in capsys.readouterr().out
    main([path, "--redact", "--format", "json"])
    assert "before" not in json.loads(capsys.readouterr().out)["breaks"][0]


def test_fail_under(capsys):
    path = str(EXAMPLES / "timestamp_in_system.jsonl")
    assert main([path, "--fail-under", "0.5"]) == EXIT_BELOW_THRESHOLD
    healthy = str(EXAMPLES / "healthy.jsonl")
    assert main([healthy, "--fail-under", "0.5"]) == EXIT_OK


def test_input_price_is_applied(capsys):
    path = str(EXAMPLES / "timestamp_in_system.jsonl")
    default = json_report(capsys, path)["estimated_waste_usd"]
    doubled = json_report(capsys, path, "--input-price", "8")["estimated_waste_usd"]
    assert doubled == pytest.approx(default * 2)


def test_usage_errors(tmp_path, capsys):
    assert main([str(tmp_path / "nope.jsonl")]) == EXIT_USAGE
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"messages": []}\n{oops\n')
    assert main([str(bad)]) == EXIT_USAGE
    assert "line 2" in capsys.readouterr().err
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    assert main([str(empty)]) == EXIT_USAGE


def test_no_usage_is_reported_plainly(tmp_path, capsys):
    path = tmp_path / "plain.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "x"}]}\n')
    assert main([str(path)]) == EXIT_OK
    assert "no usage data" in capsys.readouterr().out
