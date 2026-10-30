# prompt-cache-lens

A cache miss is one changed byte somewhere in the prefix. This finds the byte.

Prompt caching is a prefix match over tools, then system, then messages, so a timestamp near the top of a system prompt quietly costs you every request after it. Requests keep succeeding, and the `usage` fields only say that the prefix broke, not where. This diffs consecutive requests from a log, points at the first byte that changed, and names the usual culprit.

No dependencies, no API key, no network. Not affiliated with Anthropic.

## Quickstart

```sh
pip install -e .
prompt-cache-lens examples/timestamp_in_system.jsonl
```

```text
$ prompt-cache-lens examples/timestamp_in_system.jsonl
examples/timestamp_in_system.jsonl: 6 requests, 5 consecutive pairs
hit rate: 0.0% of 5,472 prompt tokens (read 0, written 5,472, uncached 0)
breaks: timestamp x5
estimated waste: 5,370 base-input-token equivalents (about $0.0215)

lines 1 -> 2: timestamp: a timestamp changed inside the prefix
  first changed byte: 798 of 3,308, at system[0].text
  2026-10-30T09:00:24Z -> 2026-10-30T09:01:01Z
  before: text":"Current time: 2026-10-30T09:00:24Z\nYou are the Example D
  after:  text":"Current time: 2026-10-30T09:01:01Z\nYou are the Example D
                                              ^
  fix: Keep the prefix frozen. Move the time to the end of `messages`, or to a mid-conversation system message where the model supports it. (shared/prompt-caching.md, Silent invalidators)
  (the same cause in the same section on 4 more pair(s))
```

The same conversation without the timestamp:

```text
$ prompt-cache-lens examples/healthy.jsonl
examples/healthy.jsonl: 6 requests, 5 consecutive pairs
hit rate: 81.8% of 5,418 prompt tokens (read 4,430, written 988, uncached 0)
no prefix breaks found
```

**Every example here is synthetic.** `scripts/make_examples.py` generates the prompts and seeded identifiers, and a toy prefix cache in that script computes the `usage` blocks, counting tokens as bytes divided by 4. The numbers show what the output looks like. They are not measurements of the API, and this has not been run on real traffic.

## What it names

Differences in the bytes:

| Cause | Meaning |
|---|---|
| `timestamp` | a date, time or epoch value differs at the first changed byte |
| `request_id` | a UUID, long hex string or named ID such as `req_...` differs there |
| `unsorted_tools` | the same tools arrived in a different order |
| `json_key_order` | same content, keys serialized in a different order |
| `tool_set_changed`, `tool_edited` | a tool was added, removed or changed |
| `system_edited` | other system prompt edits, with a hint if the new text holds a date or ID |
| `history_rewritten`, `history_truncated` | an earlier message was edited or dropped |
| `new_conversation`, `model_switch` | expected boundaries, not counted as waste |

Stable bytes, but the usage shows a miss:

| Cause | Meaning |
|---|---|
| `no_breakpoint` | nothing marks a cache breakpoint |
| `below_minimum` | shorter than the model's minimum cacheable size |
| `ttl_expired` | the gap between request starts exceeded the TTL (needs a `timestamp` field) |
| `params_changed` | `tool_choice`, `thinking` or `effort` changed |
| `lookback_exceeded` | the turn added more than 20 positions |
| `unexplained` | none of the above: concurrency, split workspaces, or something server side |

`cache_control` markers are stripped before comparing, because a marker that moves is not an invalidator.

Key order, for instance:

```text
$ prompt-cache-lens examples/varying_key_order.jsonl
examples/varying_key_order.jsonl: 6 requests, 5 consecutive pairs
hit rate: 64.2% of 5,418 prompt tokens (read 3,476, written 1,942, uncached 0)
breaks: json_key_order x5
estimated waste: 1,145 base-input-token equivalents (about $0.0046)

lines 1 -> 2: json_key_order: the same content was serialized with its keys in a different order
  first changed byte: 390 of 3,272, at tools[1].input_schema.required
  in tools[1]
  before: desk needs topic.","input_schema":{"type":"object","properties":
  after:  desk needs topic.","input_schema":{"required":["topic"],"propert
                                              ^
  fix: Serialize with sorted keys, for example `json.dumps(d, sort_keys=True)`, and do not build objects from sets. (shared/prompt-caching.md, Silent invalidators)
  (the same cause in the same section on 4 more pair(s))
```

A regression, showing the turn where it started:

```text
$ prompt-cache-lens examples/regression_mid_session.jsonl
examples/regression_mid_session.jsonl: 6 requests, 5 consecutive pairs
hit rate: 30.7% of 5,433 prompt tokens (read 1,670, written 3,763, uncached 0)
breaks: timestamp x2, system_edited x1
estimated waste: 3,324 base-input-token equivalents (about $0.0133)

lines 3 -> 4: system_edited: the system prompt text changed
  first changed byte: 769 of 3,544, at system[0].text
  the new text contains what looks like a timestamp: 2026-11-03
  before: "]}}\nsystem:\n{"type":"text","text":"You are the Example Desk ass
  after:  "]}}\nsystem:\n{"type":"text","text":"Today is 2026-11-03.\nYou ar
                                                ^
  fix: Move changing sections out of top-level `system`, after the cached prefix. (shared/prompt-caching.md, Invalidation hierarchy)
```

Causes and fixes draw on `shared/prompt-caching.md` in the `claude-api` skill bundled with Claude Code 2.1.309.

## Input

JSONL, one request per line. Log the full request body you actually sent:

```json
{"timestamp": "2026-10-30T09:00:00Z", "request": {"model": "...", "tools": [], "system": "...", "messages": []}, "usage": {"input_tokens": 12, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 900}}
```

A flat body with `usage` beside it, and `response.usage`, are also read. Without usage you still get the prefix analysis. If conversations are interleaved, separate them with `--group-by metadata.conversation_id`, otherwise every neighbouring pair looks like a break.

## Hit rate and estimated waste

Hit rate is `cache_read_input_tokens / (input_tokens + cache_creation_input_tokens + cache_read_input_tokens)`, summed over the file. The first request of a session always counts as a miss.

Waste is an estimate for each break: the tokens of the previous prompt that were not read, capped at what this request paid above the read rate, priced as the gap between what was paid (write 1.25x or 2x, uncached 1x) and the read rate (0.1x, 0.05x on Opus 5.5, 0.025x on Fable 5.1 and Mythos 5.1). It is reported in base-input-token equivalents, and in dollars when the model's price is in the table or `--input-price` is passed. Prices and multipliers were transcribed on 2026-10-30, and will go stale. Treat the result as a rough size, not a bill: it ignores the write you would still pay once after a fix.

For CI, `--fail-under 0.5` exits 1 below that hit rate, and `--format json` gives the same data. `--redact` leaves prompt excerpts out of the output, since logs tend to hold real content.

```text
$ prompt-cache-lens examples/request_id_in_system.jsonl --redact --format json   # breaks trimmed to one here
{
  "version": 1,
  "requests": 6,
  "pairs": 5,
  "hit_rate": 0.0,
  "prompt_tokens": 5490,
  "cache_read_input_tokens": 0,
  "cache_creation_input_tokens": 5490,
  "input_tokens": 0,
  "causes": {
    "request_id": 5
  },
  "estimated_waste_base_tokens": 5388.0,
  "estimated_waste_usd": 0.021551999999999998,
  "facts_written": "2026-10-30",
  "breaks": [
    {
      "previous_line": 1,
      "line": 2,
      "group": "",
      "cause": "request_id",
      "summary": "a per-request identifier changed inside the prefix",
      "fix": "Remove the ID from the prompt, or move it after the last breakpoint. Log it instead. (shared/prompt-caching.md, Silent invalidators)",
      "first_changed_byte": 781,
      "previous_bytes": 3322,
      "bytes": 3458,
      "path": "system[0].text",
      "detail": "8f6e2a87-c60d-5e53-357a-dacabc9ae844 -> 5b062361-9d83-609a-439e-f482cee95ac0",
      "cache_read_input_tokens": 0,
      "estimated_wasted_tokens": 830,
      "estimated_waste_base_tokens": 996.0,
      "estimated_waste_usd": 0.003984
    }
  ]
}
```

## Limits

- The byte offsets are into this tool's own rendering, which follows the request's order and logged key order. The API's real rendering is server side and not visible. Two requests that differ here will normally differ there, but the offset is a location in a stand-in, not a position the API would report. A string and a one-element text block are treated as equal here.
- It compares each request with the one before it. A request can read an entry written earlier than that, so a "break" can still show reads, and the usage numbers stay the ground truth.
- Cause names are pattern matches at the first difference. Only the first difference is examined; a second invalidator further along appears once the first is fixed. A monotonically changing counter that matches no pattern comes out as `system_edited` or `history_rewritten`.
- Server-side invalidations (thinking blocks stripped on some models, a lookback miss with identical payloads, concurrent requests) leave no trace in the bytes. Those appear as `unexplained` or are missed.
- Thresholds are from the documentation and not verified against the API: the minimum sizes, the TTLs and the lookback window.
- Tests are self-written fixtures: `74 passed in 0.07s`. About 954 lines of source, plus the generator.

## License

MIT
