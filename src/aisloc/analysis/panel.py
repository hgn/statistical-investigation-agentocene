"""Consolidate JSONL records into tidy panels (CSV + Parquet).

Outputs in data/panels/:
  * repo-meta.{csv,parquet}   -- one row per repo: summary, suitability,
                                 inclusion verdict, signature flags
  * repo-month.{csv,parquet}  -- repo x month x language source churn (main panel)
  * repo-activity.parquet     -- repo x month commits/authors/active_days
  * author-month.parquet      -- developer x month churn (feeds the propensity model)

These are the "clean dataset ready for R/Pandas" the brief asks for; the stats
module consumes only these, never the raw records.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..mining.language import suitability
from .frameio import save_table
from .inclusion import InclusionRule, evaluate
from .records import iter_records, summarise


def build_frames(records_dir: Path, rule: InclusionRule) -> dict[str, pd.DataFrame]:
    meta_rows: list[dict] = []
    month_rows: list[dict] = []
    act_rows: list[dict] = []
    author_rows: list[dict] = []
    mention_rows: list[dict] = []

    for rec in iter_records(records_dir):
        rid = str(rec.get("repo_id"))
        s = summarise(rec)
        v = evaluate(s, rule)

        # churn-weighted suitability across the repo's languages
        wsum = wtot = 0.0
        for m in rec.get("months") or []:
            churn = int(m["ins"]) + int(m["del"])
            wsum += suitability(str(m["lang"])) * churn
            wtot += churn
        suit_weighted = (wsum / wtot) if wtot else suitability(s.languages[0] if s.languages else "")

        primary = (rec.get("meta") or {}).get("primary_language") or (
            s.languages[0] if s.languages else "other"
        )
        meta_rows.append({
            "repo_id": rid, "provider": s.provider, "name": s.name, "url": s.url,
            "primary_language": str(primary).lower(),
            "suitability_primary": suitability(str(primary).lower()),
            "suitability_weighted": round(suit_weighted, 4),
            "contributors": s.contributors, "commits": s.commits,
            "source_added": s.source_sloc_added, "source_deleted": s.source_sloc_deleted,
            "first_month": s.first_month, "last_month": s.last_month,
            "span_months": s.span_months, "active_months": s.active_months,
            "max_gap_months": s.max_gap_months,
            "included": v.included, "exclusion_reasons": "; ".join(v.reasons),
            "has_signature": bool(s.ai_signatures),
            "first_ai_month": s.first_ai_month or "",
            "ai_author_share": round(s.ai_author_share, 4),
            "stars": (rec.get("meta") or {}).get("stars"),
        })
        for m in rec.get("months") or []:
            ins, dele = int(m["ins"]), int(m["del"])
            month_rows.append({
                "repo_id": rid, "ym": str(m["ym"]), "lang": str(m["lang"]),
                "ins": ins, "del": dele, "churn": ins + dele, "net": ins - dele,
                "files": int(m["files"]), "commits": int(m["commits"]),
                "suitability": suitability(str(m["lang"])),
            })
        for a in rec.get("activity") or []:
            act_rows.append({
                "repo_id": rid, "ym": str(a["ym"]), "commits": int(a["commits"]),
                "authors": int(a["authors"]), "active_days": int(a["active_days"]),
            })
        sig_devs = set((rec.get("signatures") or {}).get("authors") or {})
        style = rec.get("style") or {}
        for a in rec.get("authors") or []:
            ins, dele = int(a["ins"]), int(a["del"])
            dev = str(a["dev"])
            ym = str(a["ym"])
            feats = (style.get(dev) or {}).get(ym) or {}
            author_rows.append({
                "repo_id": rid, "dev": dev, "ym": ym,
                "ins": ins, "del": dele, "churn": ins + dele,
                "commits": int(a["commits"]), "n_langs": len(a.get("langs") or []),
                "dev_has_sig": dev in sig_devs,
                "size_mean": float(a.get("size_mean", 0.0)),
                "size_std": float(a.get("size_std", 0.0)),
                "size_max": int(a.get("size_max", 0)),
                "weekend_share": float(a.get("weekend_share", 0.0)),
                "evening_share": float(a.get("evening_share", 0.0)),
                "msg_chars": float(feats.get("chars", 0.0)),
                "msg_words": float(feats.get("words", 0.0)),
                "msg_lines": float(feats.get("lines", 0.0)),
                "msg_has_bullets": float(feats.get("has_bullets", 0.0)),
                "msg_unique_word_ratio": float(feats.get("unique_word_ratio", 0.0)),
                "bugfix_commits": int(feats.get("bugfix", 0)),
                "feature_commits": int(feats.get("feature", 0)),
                "total_commits": int(feats.get("total", 0)),
            })

        # Dedicated, standalone "hype curve": raw bare-word AI-tool mentions
        # per (repo, month), deliberately separate from the strict Tier-1
        # signature detection above -- see mining/stylometry.py's `mentions()`.
        # Never merged into author-month/p_ai; report as its own descriptive
        # time series only.
        for ym, counts in (rec.get("mentions") or {}).items():
            total = int(counts.get("total", 0))
            row = {"repo_id": rid, "ym": str(ym), "total_commits": total}
            for term in ("claude", "copilot", "chatgpt", "gpt", "cursor", "cody",
                        "tabnine", "codeium", "windsurf", "aider", "ai_generic"):
                row[f"mentions_{term}"] = int(counts.get(term, 0))
            mention_rows.append(row)

    frames = {
        "repo-meta": pd.DataFrame(meta_rows),
        "repo-month": pd.DataFrame(month_rows),
        "repo-activity": pd.DataFrame(act_rows),
        "author-month": pd.DataFrame(author_rows),
        "mentions-month": pd.DataFrame(mention_rows),
    }
    # month index relative to the anchor: negative = pre-AI, 0 = anchor month
    anchor_ord = _ord(rule.anchor)
    for key in ("repo-month", "repo-activity", "author-month", "mentions-month"):
        df = frames[key]
        if not df.empty:
            df["t"] = df["ym"].map(_ord) - anchor_ord
    return frames


def _ord(ym: str) -> int:
    y, m = ym.split("-")
    return int(y) * 12 + (int(m) - 1)


def write_frames(frames: dict[str, pd.DataFrame], out_dir: Path) -> None:
    for name, df in frames.items():
        # CSV for the small/headline tables; parquet-if-available for all.
        save_table(df, out_dir, name, csv=name in ("repo-meta", "repo-month", "mentions-month"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aisloc.analysis.panel", description=__doc__)
    p.add_argument("--records", default="data/records", type=Path)
    p.add_argument("--out", default="data/panels", type=Path)
    p.add_argument("--anchor", default=InclusionRule().anchor)
    a = p.parse_args(argv)
    frames = build_frames(a.records, InclusionRule(anchor=a.anchor))
    write_frames(frames, a.out)
    meta = frames["repo-meta"]
    inc = int(meta["included"].sum()) if not meta.empty else 0
    print(
        f"[panel] {len(meta)} repos ({inc} included) | "
        f"{len(frames['repo-month'])} repo-month-lang rows | "
        f"{len(frames['author-month'])} author-month rows -> {a.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
