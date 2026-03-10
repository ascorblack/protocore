"""Small JSON-oriented helpers shared across core modules."""

from __future__ import annotations


def structured_json_candidates(raw_text: str) -> list[str]:
    """Generate best-effort JSON candidates from raw model output.

    The helper accepts plain JSON, fenced JSON blocks, and noisy responses that
    wrap the JSON object/array with explanatory text.
    """

    text = raw_text.strip()
    candidates: list[str] = []
    if text:
        candidates.append(text)

    if text.startswith("```") and text.endswith("```"):
        inner = text[3:-3].strip()
        first_newline = inner.find("\n")
        if first_newline != -1:
            language = inner[:first_newline].strip().lower()
            if language in {"json", "javascript", "js"}:
                inner = inner[first_newline + 1 :].strip()
        elif inner.lower().startswith("json"):
            inner = inner[4:].strip()
        if inner:
            candidates.append(inner)

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        candidates.append(text[first_obj:last_obj + 1].strip())

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and first_arr < last_arr:
        candidates.append(text[first_arr:last_arr + 1].strip())

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique
