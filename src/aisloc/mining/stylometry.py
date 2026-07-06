"""Commit-message stylometry: lightweight structural features per commit.

Aggregated per (author, month), these give a behavioral fingerprint that is
independent of any trailer or self-disclosure: AI coding assistants tend to
produce characteristically longer, more structured (bulleted/listed) commit
messages than a given individual's own historical style. A structural break in
these features around a person's own baseline, rather than a gradual drift, is
suggestive of a tool-driven shift -- descriptive signal for Design E's
propensity model, never proof on its own (a person could simply change their
own habits for unrelated reasons).
"""

from __future__ import annotations

import re

_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']*")


def message_features(body: str) -> dict[str, float]:
    """Per-commit structural features from a commit message body (subject +
    body, as returned by git's ``%B``)."""
    words = _WORD_RE.findall(body)
    n_words = len(words)
    unique_ratio = (len({w.lower() for w in words}) / n_words) if n_words else 0.0
    return {
        "chars": float(len(body)),
        "words": float(n_words),
        "lines": float(body.count("\n") + 1),
        "has_bullets": 1.0 if _BULLET_RE.search(body) else 0.0,
        "unique_word_ratio": unique_ratio,
    }


FEATURE_KEYS = ("chars", "words", "lines", "has_bullets", "unique_word_ratio")


# Conservative, conventional-commit-aware keyword classification: does a given
# commit read as a bugfix or as new-feature work? Neither pattern matches many
# commits (docs, chores, refactors) -- that's intentional, "unclassified" is a
# valid outcome and better than a low-precision guess for either bucket.
_BUGFIX_RE = re.compile(r"(?i)^(?:fix|bugfix|hotfix|patch)(?:\(|:|!)|(?<!\w)\bfix(?:e[sd])?\b")
_FEATURE_RE = re.compile(r"(?i)^(?:feat|feature)(?:\(|:|!)|\b(?:add|implement|introduce)\b")


def classify_commit(body: str) -> str | None:
    """First line only (the subject), the conventional place intent is stated.
    Returns "bugfix", "feature", or None (unclassified)."""
    subject = body.split("\n", 1)[0]
    if _BUGFIX_RE.search(subject):
        return "bugfix"
    if _FEATURE_RE.search(subject):
        return "feature"
    return None


# Deliberately loose, bare-word mentions -- a separate, dedicated signal from
# the strict Tier-1 attribution patterns in signatures.py, which require
# attribution *context* ("Co-Authored-By:", "Generated with") for precision.
# Here we want the opposite: catch a plain "used claude for this" or "cursor
# helped debug this" that would never match the strict patterns. This is a
# raw awareness/mention-rate time series, not a detection mechanism -- report
# it as its own descriptive chart, never folded into the p_ai/propensity model.
#
# Some of these words are otherwise-ordinary English words or personal names
# ("cursor" is a routine terminal/DB/editor term; "cody" and "claude" are
# common given names; "aider" is French for "to help"; "windsurf" is a sport).
# Confirmed against real data: pallets/click (a CLI framework, 2019, years
# before the Cursor IDE existed) matched bare "cursor" from ordinary
# text-cursor-positioning code. Counting these unconditionally at 5000-repo
# scale would drown the signal in false positives. High-precision terms count
# on a bare match; ambiguous ones only count if an AI-context qualifier word
# also appears anywhere in the same commit body.
_HIGH_PRECISION_TERMS: dict[str, re.Pattern[str]] = {
    "claude": re.compile(r"\bclaude\b", re.I),
    "copilot": re.compile(r"\bcopilot\b", re.I),
    "chatgpt": re.compile(r"\bchatgpt\b", re.I),
    "gpt": re.compile(r"\bgpt-?[34o]\b", re.I),
    "tabnine": re.compile(r"\btabnine\b", re.I),
    "codeium": re.compile(r"\bcodeium\b", re.I),
    "ai_generic": re.compile(r"\bai[- ]generated\b|\bwith ai\b|\busing ai\b", re.I),
}
_AMBIGUOUS_TERMS: dict[str, re.Pattern[str]] = {
    "cursor": re.compile(r"\bcursor\b", re.I),
    "cody": re.compile(r"\bcody\b", re.I),
    "windsurf": re.compile(r"\bwindsurf\b", re.I),
    "aider": re.compile(r"\baider\b", re.I),
}
_AI_CONTEXT_RE = re.compile(
    r"\b(?:ai|llm|assistant|autocomplete|ide|chatbot|prompt|generat(?:e[ds]?|ing|ion))\b", re.I
)
_MENTION_TERMS: dict[str, re.Pattern[str]] = {**_HIGH_PRECISION_TERMS, **_AMBIGUOUS_TERMS}


def mentions(body: str) -> frozenset[str]:
    """Which AI-tool terms are literally mentioned anywhere in this commit
    message, regardless of attribution context. Ambiguous terms (common words
    or names) require an AI-context qualifier elsewhere in the same body."""
    found = {term for term, pat in _HIGH_PRECISION_TERMS.items() if pat.search(body)}
    if _AI_CONTEXT_RE.search(body):
        found |= {term for term, pat in _AMBIGUOUS_TERMS.items() if pat.search(body)}
    return frozenset(found)
