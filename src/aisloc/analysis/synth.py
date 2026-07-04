"""Synthetic record generator with a known, planted AI effect.

Writes records in the exact JSONL schema the gatherer emits, so the whole
analysis chain (panel -> stats -> plots) can be validated against ground truth:

* dose-response: post-adoption churn is multiplied by (1 + BETA * suitability),
  so the effect is large for high-AI-suitability languages and ~0 for niche ones;
* staggered adoption around the 2022-11 shock, a fraction never adopting;
* PU structure: only some adopters leave a *surviving* signature (positive
  label); the rest are censored, exactly the real-world problem.

Ground-truth parameters are also written to data/results/synth-truth.json so
stats output can be checked against them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..mining.language import suitability
from .records import ym_ord

# Languages spanning the AI-suitability range (so dose-response is estimable).
_LANGS = ["python", "typescript", "go", "java", "cpp", "c", "nix", "ada", "verilog", "cobol"]

BETA = 0.6  # planted dose-response slope: mult = 1 + BETA * suitability
ANCHOR = "2022-11"
SIG_SURVIVAL = 0.35  # fraction of adopters leaving a visible signature


def _months(start: str, end: str) -> list[str]:
    a, b = ym_ord(start), ym_ord(end)
    return [f"{o // 12:04d}-{o % 12 + 1:02d}" for o in range(a, b + 1)]


def _gen_repo(idx: int, rng: np.random.Generator, months: list[str]) -> tuple[dict, dict]:
    lang = _LANGS[idx % len(_LANGS)]
    suit = suitability(lang)
    n_authors = int(rng.integers(3, 9))
    devs = [f"dev{idx}-{k:02d}" for k in range(n_authors)]

    adopts = rng.random() < 0.65
    adopt_ord = ym_ord(ANCHOR) + int(rng.integers(0, 20)) if adopts else 10**9
    base_level = float(rng.uniform(200, 1200))  # repo-specific monthly churn scale
    trend = float(rng.uniform(-0.002, 0.004))
    leaves_sig = adopts and rng.random() < SIG_SURVIVAL

    repo_months: list[dict] = []
    activity: list[dict] = []
    authors: list[dict] = []
    sig_repo: dict[str, str] = {}
    sig_authors: dict[str, dict[str, str]] = {}
    first_adopt_month: str | None = None

    for t, ym in enumerate(months):
        post = ym_ord(ym) >= adopt_ord
        mult = (1.0 + BETA * suit) if post else 1.0
        season = 1.0 + 0.15 * np.sin(2 * np.pi * (t % 12) / 12)
        level = base_level * (1 + trend) ** t * season * mult
        ins = int(max(0, rng.normal(level, level * 0.25)))
        dele = int(max(0, rng.normal(level * 0.35, level * 0.15)))
        commits = int(max(1, rng.poisson(6 * season * (1.3 if post else 1.0))))
        if ins == 0 and commits == 0:
            continue
        if post and first_adopt_month is None:
            first_adopt_month = ym

        repo_months.append(
            {"ym": ym, "lang": lang, "ins": ins, "del": dele,
             "files": commits, "commits": commits}
        )
        activity.append(
            {"ym": ym, "commits": commits, "authors": min(n_authors, commits),
             "active_days": min(22, commits)}
        )
        # split churn across a random subset of authors this month
        active = rng.choice(devs, size=min(n_authors, max(1, commits // 2)), replace=False)
        share_ins = ins // len(active)
        for d in active:
            authors.append(
                {"ym": ym, "ins": share_ins, "del": dele // len(active),
                 "commits": max(1, commits // len(active)), "langs": [lang], "dev": d}
            )
            if post and leaves_sig and rng.random() < 0.4:
                sig_authors.setdefault(d, {}).setdefault("claude", ym)

    if leaves_sig and first_adopt_month:
        sig_repo["claude"] = first_adopt_month

    record = {
        "schema": 1, "provider": "synth", "repo_id": f"synth-{idx:04d}",
        "name": f"synth/repo-{idx:04d}", "clone_url": f"https://synth.local/repo-{idx:04d}.git",
        "default_branch": "main", "gathered_at": "2026-07-04T00:00:00+00:00",
        "baseline_since": "2019-01-01", "shallow": True,
        "meta": {"primary_language": lang, "stars": int(rng.integers(50, 5000))},
        "signatures": {"repo": sig_repo, "authors": sig_authors},
        "months": repo_months, "activity": activity, "authors": authors, "status": "ok",
    }
    truth = {"repo_id": record["repo_id"], "lang": lang, "suitability": suit,
             "adopts": adopts, "adopt_month": (None if not adopts else
             f"{adopt_ord // 12:04d}-{adopt_ord % 12 + 1:02d}"), "leaves_sig": leaves_sig}
    return record, truth


def generate(n: int, out_dir: Path, results_dir: Path, seed: int) -> int:
    rng = np.random.default_rng(seed)
    months = _months("2019-01", "2025-12")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    truths = []
    path = out_dir / "records-synth-000.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            rec, truth = _gen_repo(i, rng, months)
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            truths.append(truth)
    (results_dir / "synth-truth.json").write_text(
        json.dumps({"beta": BETA, "anchor": ANCHOR, "repos": truths}, indent=2)
    )
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aisloc.analysis.synth", description=__doc__)
    p.add_argument("-n", "--repos", type=int, default=80)
    p.add_argument("--records", default="data/records", type=Path)
    p.add_argument("--results", default="data/results", type=Path)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    n = generate(a.repos, a.records, a.results, a.seed)
    print(f"[synth] wrote {n} synthetic repos (beta={BETA}, anchor={ANCHOR})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
