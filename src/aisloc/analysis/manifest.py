"""Per-repo audit manifest.

Lists every examined repository by full URL, with indented facts beneath it:
whether it was used for the analysis (and why not), contributor count, the
(preliminary) probability that AI was adopted, and other relevant fields.

Two outputs:
  * data/results/repo-manifest.txt  -- human-readable, indented
  * data/results/repo-manifest.csv  -- machine-readable, one row per repo

The probability here is PRELIMINARY and signature-based: present signature =>
high confidence; absent => unknown (left to the PU propensity model in
`aisloc.analysis.stats`, Design E). Absence of a signature is NOT evidence of no
AI (trailers get stripped), so we never emit a low probability for "no signal".
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .inclusion import InclusionRule, Verdict, evaluate
from .records import RepoSummary, iter_records, summarise


def prelim_p_ai(s: RepoSummary) -> float | None:
    """Preliminary P(repo adopted AI). Only defined where a signature survives;
    None means 'unknown, pending PU model' (never a low number)."""
    if not s.ai_signatures:
        return None
    # More surviving signal classes / higher author share -> firmer.
    base = 0.90 + 0.03 * min(len(s.ai_signatures), 3)
    return round(min(0.99, base + 0.05 * s.ai_author_share), 3)


def load_model_propensity(out_dir: Path) -> dict[str, dict[str, str]]:
    """Load calibrated per-repo P(AI) from the PU model if stats has been run."""
    path = out_dir / "propensity-repo.csv"
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["repo_id"]] = row
    return out


def resolve_p_ai(s: RepoSummary, model: dict[str, dict[str, str]]) -> tuple[float | None, str]:
    """Prefer the calibrated model probability; fall back to the preliminary one."""
    m = model.get(s.repo_id)
    if m and m.get("p_ai") not in (None, "", "nan"):
        lo, hi = m.get("p_ai_lo", ""), m.get("p_ai_hi", "")
        ci = f" [{float(lo):.2f}, {float(hi):.2f}]" if lo not in ("", "nan") else ""
        return float(m["p_ai"]), f"{float(m['p_ai']):.3f}{ci} ({m.get('label_source', 'model')})"
    p = prelim_p_ai(s)
    return p, (f"{p:.3f} (preliminary: signature)" if p is not None else "n/a (pending PU model)")


CSV_FIELDS = [
    "url", "provider", "repo_id", "name", "used_for_analysis", "exclusion_reasons",
    "contributors", "commits", "source_sloc_added", "source_sloc_deleted",
    "first_month", "last_month", "span_months", "active_months", "max_gap_months",
    "languages", "ai_signatures", "first_ai_month", "ai_author_share", "p_ai", "p_ai_detail",
]


def _row(s: RepoSummary, v: Verdict, p: float | None, p_disp: str) -> dict[str, object]:
    return {
        "url": s.url,
        "provider": s.provider,
        "repo_id": s.repo_id,
        "name": s.name,
        "used_for_analysis": "yes" if v.included else "no",
        "exclusion_reasons": "; ".join(v.reasons),
        "contributors": s.contributors,
        "commits": s.commits,
        "source_sloc_added": s.source_sloc_added,
        "source_sloc_deleted": s.source_sloc_deleted,
        "first_month": s.first_month,
        "last_month": s.last_month,
        "span_months": s.span_months,
        "active_months": s.active_months,
        "max_gap_months": s.max_gap_months,
        "languages": ",".join(s.languages),
        "ai_signatures": ",".join(s.ai_signatures),
        "first_ai_month": s.first_ai_month or "",
        "ai_author_share": round(s.ai_author_share, 3),
        "p_ai": "" if p is None else p,
        "p_ai_detail": p_disp,
    }


def _write_text(rows: list[tuple[RepoSummary, Verdict, float | None, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# AI-SLOC repository manifest\n")
        fh.write("# p_ai is PRELIMINARY (signature-based); absence of signal != no AI.\n")
        fh.write("# The calibrated probability is produced by `make analyze` (PU model).\n\n")
        for s, v, p, p_disp in rows:
            fh.write(f"{s.url}\n")
            fh.write(f"    used_for_analysis: {'yes' if v.included else 'no'}\n")
            if v.reasons:
                fh.write(f"    exclusion_reasons: {'; '.join(v.reasons)}\n")
            fh.write(f"    contributors:      {s.contributors}\n")
            fh.write(f"    commits:           {s.commits}\n")
            fh.write(
                f"    span:              {s.first_month}..{s.last_month} "
                f"({s.span_months}m span, {s.active_months} active, "
                f"max gap {s.max_gap_months}m)\n"
            )
            fh.write(
                f"    source_sloc:       +{s.source_sloc_added} / -{s.source_sloc_deleted}\n"
            )
            fh.write(f"    languages:         {', '.join(s.languages) or '-'}\n")
            fh.write(
                f"    ai_signatures:     {', '.join(s.ai_signatures) or 'none visible'}"
                f"{f' (first {s.first_ai_month})' if s.first_ai_month else ''}\n"
            )
            fh.write(f"    ai_author_share:   {s.ai_author_share:.2f}\n")
            fh.write(f"    p_ai:              {p_disp}\n\n")


def build(records_dir: Path, out_dir: Path, rule: InclusionRule) -> tuple[int, int]:
    model = load_model_propensity(out_dir)
    rows: list[tuple[RepoSummary, Verdict, float | None, str]] = []
    for rec in iter_records(records_dir):
        s = summarise(rec)
        v = evaluate(s, rule)
        p, p_disp = resolve_p_ai(s, model)
        rows.append((s, v, p, p_disp))
    rows.sort(key=lambda r: (not r[1].included, -r[0].contributors, r[0].name))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_text(rows, out_dir / "repo-manifest.txt")
    with (out_dir / "repo-manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for s, v, p, p_disp in rows:
            w.writerow(_row(s, v, p, p_disp))

    included = sum(1 for _s, v, _p, _d in rows if v.included)
    return len(rows), included


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aisloc.analysis.manifest", description=__doc__)
    p.add_argument("--records", default="data/records", type=Path)
    p.add_argument("--out", default="data/results", type=Path)
    p.add_argument("--anchor", default=InclusionRule().anchor, help="AI cutoff YYYY-MM")
    a = p.parse_args(argv)
    total, included = build(a.records, a.out, InclusionRule(anchor=a.anchor))
    print(
        f"[manifest] {total} repos -> {included} included, {total - included} excluded; "
        f"wrote {a.out}/repo-manifest.{{txt,csv}}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
