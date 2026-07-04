"""Parse ``git log --numstat`` into monthly panels.

Aggregates three views the analysis needs (concept.md sec. 4, 2.4b):
  * ``months``   : per (year-month, language) source churn -> the main outcome
  * ``activity`` : per year-month repo activity (commits, authors, active days)
  * ``authors``  : per (author, year-month) churn -> enables the per-user
                   propensity model (Design E)

Parsing is streaming and commit-boundary flushed, so memory stays O(months x
languages + authors x months), never O(commits).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from . import language, pathfilter

# Commit header sentinel + unit separator; chosen so they never occur in paths.
REC = "\x01"
SEP = "\x1f"

# Pretty format producing one sentinel-prefixed header line per commit, then the
# numstat block. Author email hashed downstream; kept raw here for aggregation.
LOG_PRETTY = f"{REC}%H{SEP}%aI{SEP}%ae{SEP}%an"


def log_args(since: str) -> list[str]:
    return [
        "--no-merges",
        f"--since={since}",
        "--numstat",
        f"--pretty=format:{LOG_PRETTY}",
    ]


_RENAME_BRACE = re.compile(r"\{[^{}]* => ([^{}]*)\}")


def _resolve_path(raw: str) -> str:
    """Reduce numstat rename notation to the destination path."""
    p = _RENAME_BRACE.sub(r"\1", raw)
    if " => " in p:
        p = p.split(" => ", 1)[1]
    return p.replace("//", "/").strip()


@dataclass
class _Bucket:
    ins: int = 0
    dele: int = 0
    files: int = 0
    commits: int = 0


@dataclass
class Aggregator:
    # (ym, lang) -> bucket
    lang: dict[tuple[str, str], _Bucket] = field(default_factory=lambda: defaultdict(_Bucket))
    # ym -> activity
    act_commits: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    act_authors: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    act_days: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # (email, ym) -> bucket, plus langs touched and days
    author: dict[tuple[str, str], _Bucket] = field(default_factory=lambda: defaultdict(_Bucket))
    author_langs: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    # in-progress commit state
    _ym: str = ""
    _date: str = ""
    _email: str = ""
    _langs_this_commit: set[str] = field(default_factory=set)
    _authors_this_commit: bool = False

    def start_commit(self, header: str) -> None:
        self._flush_commit()
        try:
            _hash, iso, email, _name = header.split(SEP, 3)
        except ValueError:
            self._ym = ""  # malformed; skip its files
            return
        self._ym = iso[:7]
        self._date = iso[:10]
        self._email = email.lower()
        self._langs_this_commit = set()

    def add_file(self, line: str) -> None:
        if not self._ym:
            return
        parts = line.split("\t")
        if len(parts) < 3:
            return
        ins_s, del_s, raw_path = parts[0], parts[1], "\t".join(parts[2:])
        path = _resolve_path(raw_path)
        if pathfilter.is_excluded(path):
            return
        lang = language.classify(path)
        if not language.is_source(lang):
            return
        ins = 0 if ins_s == "-" else int(ins_s)
        dele = 0 if del_s == "-" else int(del_s)

        b = self.lang[(self._ym, lang)]
        b.ins += ins
        b.dele += dele
        b.files += 1

        ab = self.author[(self._email, self._ym)]
        ab.ins += ins
        ab.dele += dele
        ab.files += 1
        self.author_langs[(self._email, self._ym)].add(lang)
        self._langs_this_commit.add(lang)
        self._authors_this_commit = True

    def _flush_commit(self) -> None:
        if not self._ym or not self._authors_this_commit:
            self._authors_this_commit = False
            return
        self.act_commits[self._ym] += 1
        self.act_authors[self._ym].add(self._email)
        self.act_days[self._ym].add(self._date)
        for lang in self._langs_this_commit:
            self.lang[(self._ym, lang)].commits += 1
        self.author[(self._email, self._ym)].commits += 1
        self._authors_this_commit = False

    # -- serialisation -------------------------------------------------------

    def to_record(self) -> dict[str, object]:
        self._flush_commit()
        months = [
            {"ym": ym, "lang": lang, "ins": b.ins, "del": b.dele,
             "files": b.files, "commits": b.commits}
            for (ym, lang), b in sorted(self.lang.items())
        ]
        activity = [
            {"ym": ym, "commits": self.act_commits[ym],
             "authors": len(self.act_authors[ym]), "active_days": len(self.act_days[ym])}
            for ym in sorted(self.act_commits)
        ]
        authors = [
            {"email": email, "ym": ym, "ins": b.ins, "del": b.dele,
             "commits": b.commits, "langs": sorted(self.author_langs[(email, ym)])}
            for (email, ym), b in sorted(self.author.items())
        ]
        return {"months": months, "activity": activity, "authors": authors}


def feed_line(agg: Aggregator, raw: bytes) -> None:
    """Route one raw stdout line into the aggregator."""
    line = raw.decode("utf-8", "replace").rstrip("\n")
    if not line:
        return
    if line.startswith(REC):
        agg.start_commit(line[1:])
    else:
        agg.add_file(line)
