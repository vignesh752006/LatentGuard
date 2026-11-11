"""Task 6 — JSON-repair and retry logic for planner responses.

The LLM/VLM planner may return:
  - Valid JSON → parsed directly.
  - JSON wrapped in markdown fences (```json ... ```) → stripped and parsed.
  - Partial JSON (trailing garbage) → truncated at last valid token.
  - Non-JSON text → raises PlannerResponseError after all retries.
"""

from __future__ import annotations

import json
import re


class PlannerResponseError(Exception):
    """Raised when the planner response cannot be repaired into valid JSON."""


def repair_json(raw: str) -> dict:
    """Attempt to extract and parse a JSON object from raw planner output.

    Strategies (tried in order):
    1. Direct parse.
    2. Strip markdown code fences and retry.
    3. Locate first ``{`` and last ``}`` and parse the substring.
    4. Remove trailing commas and retry.

    Raises PlannerResponseError if all strategies fail.
    """
    if not raw or not raw.strip():
        raise PlannerResponseError("Empty response from planner")

    strategies = [
        _direct,
        _strip_markdown,
        _extract_braces,
        _remove_trailing_commas,
    ]

    for strategy in strategies:
        result = strategy(raw)
        if result is not None:
            return result

    raise PlannerResponseError(
        f"Cannot repair planner response into valid JSON. "
        f"Response (first 200 chars): {raw[:200]!r}"
    )


def _direct(raw: str) -> dict | None:
    try:
        obj = json.loads(raw.strip())
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _strip_markdown(raw: str) -> dict | None:
    # Remove ```json ... ``` or ``` ... ``` fences
    stripped = re.sub(r"```(?:json)?\s*", "", raw)
    stripped = stripped.strip().rstrip("`").strip()
    return _direct(stripped)


def _extract_braces(raw: str) -> dict | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return _direct(raw[start : end + 1])


def _remove_trailing_commas(raw: str) -> dict | None:
    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    return _extract_braces(cleaned)

