"""Pricing multipliers and cache limits, transcribed on FACTS_WRITTEN. They will go stale."""

from __future__ import annotations

FACTS_WRITTEN = "2026-10-30"
FACTS_SOURCE = (
    "claude-api skill bundled with Claude Code 2.1.309: shared/prompt-caching.md, shared/models.md"
)

WRITE_MULTIPLIER_5M = 1.25
WRITE_MULTIPLIER_1H = 2.0
READ_MULTIPLIER_DEFAULT = 0.1
READ_MULTIPLIER = {
    "claude-fable-5-1": 0.025,
    "claude-mythos-5-1": 0.025,
    "claude-opus-5-5": 0.05,
}
# Base input price in USD per million tokens. Claude Haiku 5.5 has a second, higher rate card for
# prompts over 100K tokens, which is not modelled here.
INPUT_PRICE = {
    "claude-fable-5-1": 10.0,
    "claude-mythos-5-1": 10.0,
    "claude-fable-5": 10.0,
    "claude-opus-5-5": 4.0,
    "claude-opus-5": 5.0,
    "claude-sonnet-5-5": 2.0,
    "claude-sonnet-5": 2.0,
    "claude-haiku-5-5": 0.10,
}
MINIMUM_CACHEABLE = {
    "claude-opus-5-5": 512,
    "claude-opus-5": 512,
    "claude-fable-5": 512,
    "claude-mythos-5": 512,
    "claude-fable-5-1": 512,
    "claude-mythos-5-1": 512,
    "claude-sonnet-5-5": 512,
    "claude-haiku-5-5": 512,
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-sonnet-4-5": 1024,
    "claude-opus-4-1": 1024,
    "claude-opus-4-0": 1024,
    "claude-sonnet-4-0": 1024,
    "claude-opus-4-7": 2048,
    "claude-mythos-preview": 2048,
    "claude-opus-4-6": 4096,
    "claude-opus-4-5": 4096,
    "claude-haiku-4-5": 4096,
}
TTL_5M_SECONDS = 300
TTL_1H_SECONDS = 3600
LOOKBACK_POSITIONS = 20
