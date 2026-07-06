"""Load and summarise gathered JSONL records.

Shared by every analysis module so the on-disk schema is parsed in exactly one
place. Pure stdlib (no pandas) so lightweight tools like the manifest stay cheap.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


def iter_records(records_dir: Path) -> Iterator[dict[str, object]]:
    """Yield every repo record across all shard files (skips failures-*),
    deduplicated by (provider, name).

    A repo present in both a curated list and a provider search (e.g. this
    project's Rust-community list plus a supplementary GitHub search) gets
    assigned a different repo_id per source -- ListSource uses the
    "owner/repo" name itself, GitHubSource uses GitHub's numeric id -- so it
    can slip past the gather-time repo_id-keyed dedup and get mined twice
    under two distinct repo_ids. That would inflate its weight in every
    aggregate (it counts as two "repos"). gather.py's resume/dedup keys on
    name instead (see its _load_done) to prevent this going forward; the
    dedup here catches survivors already baked into previously-gathered
    JSONL from before that fix.
    """
    seen: set[tuple[str, str]] = set()
    for path in sorted(records_dir.glob("records-*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (str(rec.get("provider")), str(rec.get("name")))
                if key in seen:
                    continue
                seen.add(key)
                yield rec


_YM_RE = re.compile(r"^\d{4}-\d{2}$")


def _clean_ym(v: str) -> str:
    """Recover a usable year-month from records gathered before the
    signatures.py inter-commit-newline fix (git's ``pretty=format:`` inserts a
    bare newline between commits; every record but the first arrived with a
    leading "\\n", which had already been baked into a truncated 7-char slice
    at gather time, e.g. "2026-0" instead of "2026-07" -- the month's last
    digit is unrecoverable from what was stored). Falls back to year-only
    rather than showing a falsely precise, truncated month."""
    v = v.lstrip("\n")
    if _YM_RE.match(v):
        return v
    return v[:4] if len(v) >= 4 and v[:4].isdigit() else v


def ym_ord(ym: str) -> int:
    """'YYYY-MM' -> months since year 0 (for gap/span arithmetic)."""
    y, m = ym.split("-")
    return int(y) * 12 + (int(m) - 1)


@dataclass
class RepoSummary:
    provider: str
    repo_id: str
    name: str
    url: str
    contributors: int
    commits: int
    source_sloc_added: int
    source_sloc_deleted: int
    first_month: str
    last_month: str
    span_months: int
    active_months: int
    max_gap_months: int
    languages: list[str]
    ai_signatures: list[str]  # signal classes present at repo level
    first_ai_month: str | None
    ai_author_share: float  # fraction of contributors with any AI signature


def summarise(rec: dict[str, object]) -> RepoSummary:
    activity = rec.get("activity") or []
    months = rec.get("months") or []
    authors = rec.get("authors") or []

    yms = sorted(a["ym"] for a in activity)  # type: ignore[index]
    first = yms[0] if yms else ""
    last = yms[-1] if yms else ""
    span = (ym_ord(last) - ym_ord(first) + 1) if yms else 0
    active = len(yms)
    max_gap = _max_gap(yms)

    ins = sum(int(m["ins"]) for m in months)  # type: ignore[index,call-overload]
    dele = sum(int(m["del"]) for m in months)  # type: ignore[index,call-overload]
    commits = sum(int(a["commits"]) for a in activity)  # type: ignore[index,call-overload]
    langs = sorted({str(m["lang"]) for m in months})  # type: ignore[index]

    devs = {str(a["dev"]) for a in authors}  # type: ignore[index]
    sig = rec.get("signatures") or {}
    repo_sig = sig.get("repo") or {}  # type: ignore[union-attr]
    author_sig = sig.get("authors") or {}  # type: ignore[union-attr]
    first_ai = min((_clean_ym(v) for v in repo_sig.values()), default=None)
    ai_devs = {d for d in author_sig}
    ai_share = (len(ai_devs) / len(devs)) if devs else 0.0

    return RepoSummary(
        provider=str(rec.get("provider")),
        repo_id=str(rec.get("repo_id")),
        name=str(rec.get("name")),
        url=str(rec.get("clone_url") or rec.get("name")),
        contributors=len(devs),
        commits=commits,
        source_sloc_added=ins,
        source_sloc_deleted=dele,
        first_month=first,
        last_month=last,
        span_months=span,
        active_months=active,
        max_gap_months=max_gap,
        languages=langs,
        ai_signatures=sorted(repo_sig.keys()),  # type: ignore[union-attr]
        first_ai_month=first_ai,
        ai_author_share=ai_share,
    )


def _max_gap(sorted_yms: list[str]) -> int:
    if len(sorted_yms) < 2:
        return 0
    ords = [ym_ord(y) for y in sorted_yms]
    return max(b - a - 1 for a, b in zip(ords, ords[1:]))
