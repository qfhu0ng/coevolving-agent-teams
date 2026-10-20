"""What the outer optimizer and reviewer may write to disk (plan v4 section 12, B1).

DeepSeek thinking mode needs each assistant turn's reasoning_content sent back on
later tool requests, so the in-memory conversation keeps it verbatim. No trace,
request, response, retry record, conversation copy or summary keeps it: every
persisted value goes through ``without_reasoning`` first, which removes the
fields from a copy and never touches the caller's object.

Only model messages lose these fields: an assistant message (any dict whose role is
"assistant"), whatever sits under a "message" or "delta" key, every element of a
"messages" list, and every element of a conversation passed with ``message=True``,
so a reply without a role is covered too (M5 review R2-3). Request settings such as
extra_body's reasoning effort, and evidence that happens to contain a key named
"reasoning", are kept as sent (M5 review minor 9).
"""
from __future__ import annotations

# Reasoning channels: the transport's assembled ones, plus the structured form some providers return.
REASONING_KEYS = frozenset({"reasoning_content", "reasoning", "reasoning_details"})
MESSAGE_KEYS = frozenset({"message", "delta", "messages"})


def without_reasoning(value, *, message: bool = False):
    """A copy of ``value`` with every model message's reasoning fields removed."""
    if isinstance(value, dict):
        strip = message or value.get("role") == "assistant"
        return {key: without_reasoning(item, message=key in MESSAGE_KEYS)
                for key, item in value.items() if not (strip and key in REASONING_KEYS)}
    if isinstance(value, (list, tuple)):
        return [without_reasoning(item, message=message) for item in value]
    return value
