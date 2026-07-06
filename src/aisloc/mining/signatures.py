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

from .stylometry import FEATURE_KEYS, classify_commit, message_features, mentions

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
    """Accumulates earliest-month-per-class (per repo and per author email),
    plus per-(author, month) commit-message stylometry -- both derived from the
    same single log pass that already reads the commit body for signature
    detection, so no extra `git log` invocation is needed for either."""

    def __init__(self, min_ym: str = "", max_ym: str = "") -> None:
        # Hard floor/ceiling on commit year-month, enforced here rather than
        # trusted to git's own date filtering -- see churn.Aggregator.min_ym/
        # max_ym for why: a merged-in old branch can leak commits from years
        # before the requested --since cutoff past both --since and
        # --since-as-filter, and a bogus system clock at commit time can push
        # one the other way (seen for real: zebra-rs/zebra-rs, dated 2106/2242).
        self.min_ym = min_ym
        self.max_ym = max_ym
        self.repo: dict[str, str] = {}
        self.by_author: dict[str, dict[str, str]] = defaultdict(dict)
        # (email, ym) -> running sums of message_features(), plus commit count
        self._style_sum: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: dict.fromkeys(FEATURE_KEYS, 0.0)
        )
        self._style_n: dict[tuple[str, str], int] = defaultdict(int)
        # (email, ym) -> {"bugfix": n, "feature": n, "total": n}
        self._commit_class: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"bugfix": 0, "feature": 0, "total": 0}
        )
        # ym -> {term: mention_count, "total": commits_scanned} -- global (not
        # per-author), a repo-level "hype curve" over calendar time. Deliberately
        # separate from the Tier-1 signature counts above (those require
        # attribution context for precision; this is a loose bare-word count
        # by design -- see stylometry.mentions).
        self._mentions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

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
        if self.min_ym and ym < self.min_ym:
            return  # older than baseline_since despite git's filter; drop
        if self.max_ym and ym > self.max_ym:
            return  # bogus future clock (seen: real commits dated 2106, 2242)
        email = email.lower()
        hay = f"{name}\n{body}"
        for cls, pats in _PATTERNS.items():
            if any(p.search(hay) for p in pats):
                self._note(cls, email, ym)
        ident = f"{email} {name.lower()}"
        for needle, cls in _BOT_AUTHORS.items():
            if cls and needle in ident:
                self._note(cls, email, ym)

        key = (email, ym)
        feats = message_features(body)
        sums = self._style_sum[key]
        for k, v in feats.items():
            sums[k] += v
        self._style_n[key] += 1

        cls = classify_commit(body)
        counts = self._commit_class[key]
        counts["total"] += 1
        if cls is not None:
            counts[cls] += 1

        mset = mentions(body)
        month_mentions = self._mentions[ym]
        month_mentions["total"] += 1
        for term in mset:
            month_mentions[term] += 1

    def to_record(self) -> dict[str, object]:
        style: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for (email, ym), sums in self._style_sum.items():
            n = self._style_n[(email, ym)]
            style[email][ym] = {k: round(v / n, 4) for k, v in sums.items()} | {"n": n}
            style[email][ym].update(self._commit_class[(email, ym)])
        return {
            "repo": dict(sorted(self.repo.items())),
            "authors": {e: dict(sorted(c.items())) for e, c in sorted(self.by_author.items())},
            "style": {e: dict(sorted(m.items())) for e, m in sorted(style.items())},
            "mentions": {ym: dict(sorted(c.items())) for ym, c in sorted(self._mentions.items())},
        }


async def scan(lines: AsyncIterator[bytes], min_ym: str = "", max_ym: str = "") -> dict[str, object]:
    """Consume raw log stdout (record-separated by RS) and return detections.

    Bodies contain newlines, so we buffer and split on the RS byte rather than
    per line, keeping at most one record in memory at a time.

    Git's ``pretty=format:`` inserts a bare newline *between* commits (though
    not after the last one), so every record except the first arrives with a
    leading ``\\n``. Left in place, that byte lands inside the %aI date field
    after the SEP split (`iso[:7]` becomes e.g. "\\n2026-0"), and since ASCII
    ``\\n`` sorts below every digit, it also silently wins any ``min()`` over
    year-month strings -- corrupting "earliest AI month" for any record but
    the very first. Strip it before parsing.
    """
    scanner = Scanner(min_ym=min_ym, max_ym=max_ym)
    buf = ""
    async for raw in lines:
        buf += raw.decode("utf-8", "replace")
        while RS in buf:
            record, buf = buf.split(RS, 1)
            scanner.scan_record(record.lstrip("\n"))
    if buf.strip():
        scanner.scan_record(buf.lstrip("\n"))
    return scanner.to_record()
