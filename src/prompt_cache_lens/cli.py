"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path

from . import __version__, facts
from .analyze import Break, InputError, Report, analyze, load_records

EXIT_OK, EXIT_BELOW_THRESHOLD, EXIT_USAGE = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-cache-lens",
        description="Find the first byte that changed between Messages API requests, and why.",
        epilog=f"Prices and limits transcribed {facts.FACTS_WRITTEN} from: {facts.FACTS_SOURCE}.",
    )
    parser.add_argument("path", type=Path, help="JSONL of requests with usage; - for stdin")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--group-by",
        metavar="FIELD",
        help="dotted field separating conversations, for example metadata.conversation_id",
    )
    parser.add_argument(
        "--input-price", type=float, metavar="USD", help="base input price per million tokens"
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        metavar="RATE",
        help="exit 1 if the hit rate is below this (0-1)",
    )
    parser.add_argument("--all", action="store_true", help="show every break, not one per run")
    parser.add_argument("--redact", action="store_true", help="omit prompt excerpts from output")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def print_break(run: list[Break], redact: bool) -> None:
    item = run[0]
    head = f"lines {item.previous.line} -> {item.current.line}"
    if item.current.group:
        head += f" [{item.current.group}]"
    print(f"{head}: {item.cause.id}: {item.cause.summary}")
    if item.offset is not None:
        print(
            f"  first changed byte: {item.offset:,} of {len(item.previous.rendered.data):,}, "
            f"at {item.path}"
        )
    if item.detail and not redact:
        print(f"  {item.detail}")
    if item.offset is not None and not redact:
        print(f"  before: {item.before}")
        print(f"  after:  {item.after}")
        print(f"          {' ' * item.caret}^")
    print(f"  fix: {item.cause.fix}")
    if len(run) > 1:
        print(f"  (the same cause in the same section on {len(run) - 1} more pair(s))")


def _section(item: Break) -> str:
    return (item.path or "").split(".")[0]


def print_text(report: Report, name: str, show_all: bool, redact: bool) -> None:
    print(f"{name}: {len(report.records)} requests, {report.pairs} consecutive pairs")
    if report.hit_rate is None:
        print("no usage data: prefix analysis only, no hit rate or waste estimate")
    else:
        print(
            f"hit rate: {report.hit_rate:.1%} of {report.prompt_tokens:,} prompt tokens "
            f"(read {report.read_tokens:,}, written {report.written_tokens:,}, "
            f"uncached {report.uncached_tokens:,})"
        )
    if not report.breaks:
        print("no prefix breaks found")
    else:
        counts = ", ".join(f"{cause} x{n}" for cause, n in report.cause_counts().most_common())
        print(f"breaks: {counts}")
        if report.hit_rate is not None:
            usd = f" (about ${report.waste_usd:.4f})" if report.waste_usd is not None else ""
            print(f"estimated waste: {report.waste_units:,.0f} base-input-token equivalents{usd}")
    if show_all:
        runs = [[item] for item in report.breaks]
    else:
        keyed = groupby(report.breaks, key=lambda b: (b.cause.id, _section(b), b.current.group))
        runs = [list(run) for _, run in keyed]
    for run in runs:
        print()
        print_break(run, redact)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        text = sys.stdin.read() if str(args.path) == "-" else args.path.read_text(encoding="utf-8")
        records = load_records(text, args.group_by)
    except OSError as exc:
        print(f"{args.path}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except InputError as exc:
        print(f"{args.path}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not records:
        print(f"{args.path}: no requests found", file=sys.stderr)
        return EXIT_USAGE

    report = analyze(records, args.input_price)
    if args.format == "json":
        payload = {
            "version": 1,
            "requests": len(report.records),
            "pairs": report.pairs,
            "hit_rate": report.hit_rate,
            "prompt_tokens": report.prompt_tokens,
            "cache_read_input_tokens": report.read_tokens,
            "cache_creation_input_tokens": report.written_tokens,
            "input_tokens": report.uncached_tokens,
            "causes": dict(report.cause_counts()),
            "estimated_waste_base_tokens": round(report.waste_units, 1),
            "estimated_waste_usd": None if report.waste_usd is None else round(report.waste_usd, 6),
            "facts_written": facts.FACTS_WRITTEN,
            "breaks": [item.to_dict(args.redact) for item in report.breaks],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_text(report, str(args.path), args.all, args.redact)

    if (
        args.fail_under is not None
        and report.hit_rate is not None
        and report.hit_rate < args.fail_under
    ):
        return EXIT_BELOW_THRESHOLD
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
