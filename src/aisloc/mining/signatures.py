"""Detect AI-attribution signatures in commit metadata.

IMPORTANT (concept.md sec. 2, 3): these are a *positive-only, censored* label,
never a treatment variable. Trailers are routinely stripped by squash/rebase/
hooks/opt-out, so "no signature" means nothing. We record, per repo and per
author, the earliest month each signal class appears, for use only as
validation (Design D) and as a positive label for the PU propensity model
(Design E).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import AsyncIterator

SEP = "\x1f"
RS = "\x1e"

# pretty format: author-date, email, name, full body, then a record separator.
LOG_PRETTY = f"%aI{SEP}%ae{SEP}%an{SEP}%B{RS}"


def log_args(since: str) -> list[str]:
    return ["--no-merges", f"--since={since}", f"--pretty=format:{LOG_PRETTY}"]


# Patterns are intentionally conservative to keep precision high. Where a bare
# tool name is ambiguous (e.g. "cursor" also means a DB/text cursor) we only
# match it in an attribution context (co-author / "generated with").
_COAUTHOR = r"co-authored-by:[^\n]*"
_GENERATED = r"(?:generated (?:with|by)|assisted by)[^\n]*"

_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "claude": [
        re.compile(r"noreply@anthropic\.com"),
        re.compile(rf"{_COAUTHOR}claude", re.I),
        re.compile(rf"{_GENERATED}claude(?: code)?", re.I),
        re.compile(r"claude code", re.I),
    ],
    "copilot": [
        re.compile(r"copilot(?:@|\[bot\]| ide)", re.I),
        re.compile(rf"{_COAUTHOR}copilot", re.I),
        re.compile(rf"{_GENERATED}(?:github )?copilot", re.I),
    ],
    "cursor": [
        re.compile(rf"{_COAUTHOR}cursor", re.I),
        re.compile(rf"{_GENERATED}cursor", re.I),
        re.compile(r"cursor\.(?:so|com)|cursoragent", re.I),
    ],
    "aider": [
        re.compile(rf"{_COAUTHOR}aider", re.I),
        re.compile(r"(?:^|\n)aider:", re.I),
        re.compile(r"aider\.chat", re.I),
    ],
    "codeium": [re.compile(r"codeium|windsurf", re.I)],
    "cody": [re.compile(r"sourcegraph cody|(?:^|\W)cody ai", re.I)],
    "devin": [re.compile(rf"{_GENERATED}devin|devin-ai|cognition labs", re.I)],
    "tabnine": [re.compile(r"tabnine", re.I)],
}

# Known bot author identities (email/name substrings) => attribute to a class.
_BOT_AUTHORS: dict[str, str] = {
    "copilot": "copilot",
    "github-actions[bot]": "",  # excluded elsewhere; listed to avoid misfires
    "cursoragent": "cursor",
    "devin-ai-integration[bot]": "devin",
    "aider": "aider",
}


class Scanner:
    """Accumulates earliest-month-per-class, per repo and per author email."""

    def __init__(self) -> None:
        self.repo: dict[str, str] = {}
        self.by_author: dict[str, dict[str, str]] = defaultdict(dict)

    def _note(self, cls: str, email: str, ym: str) -> None:
        cur = self.repo.get(cls)
        if cur is None or ym < cur:
            self.repo[cls] = ym
        a = self.by_author[email]
        if cls not in a or ym < a[cls]:
            a[cls] = ym

    def scan_record(self, record: str) -> None:
        try:
            iso, email, name, body = record.split(SEP, 3)
        except ValueError:
            return
        ym = iso[:7]
        email = email.lower()
        hay = f"{name}\n{body}"
        for cls, pats in _PATTERNS.items():
            if any(p.search(hay) for p in pats):
                self._note(cls, email, ym)
        ident = f"{email} {name.lower()}"
        for needle, cls in _BOT_AUTHORS.items():
            if cls and needle in ident:
                self._note(cls, email, ym)

    def to_record(self) -> dict[str, object]:
        return {
            "repo": dict(sorted(self.repo.items())),
            "authors": {e: dict(sorted(c.items())) for e, c in sorted(self.by_author.items())},
        }


async def scan(lines: AsyncIterator[bytes]) -> dict[str, object]:
    """Consume raw log stdout (record-separated by RS) and return detections.

    Bodies contain newlines, so we buffer and split on the RS byte rather than
    per line, keeping at most one record in memory at a time."""
    scanner = Scanner()
    buf = ""
    async for raw in lines:
        buf += raw.decode("utf-8", "replace")
        while RS in buf:
            record, buf = buf.split(RS, 1)
            scanner.scan_record(record)
    if buf.strip():
        scanner.scan_record(buf)
    return scanner.to_record()
