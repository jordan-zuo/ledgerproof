from __future__ import annotations

import re

# Capitalised words that start a sentence or a question, not vendors.
_ENTITY_STOP = {
    "answer", "find", "post", "which", "what", "who", "how", "when", "where",
    "compare", "tell", "give", "does", "do", "can", "please", "return", "and",
    "but", "if", "the", "this", "that", "is", "are", "was", "were", "yes", "no",
}
_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z&]*\b")


def task_entities(task: str) -> list[str]:
    """Names a task asks for, ignoring sentence starts and months."""
    seen: list[str] = []
    for match in _ENTITY_RE.finditer(task):
        if match.start() == 0:
            continue
        term = match.group(0).lower()
        if term in _ENTITY_STOP or term in _MONTH_WORDS or term in seen:
            continue
        seen.append(term)
    return seen
