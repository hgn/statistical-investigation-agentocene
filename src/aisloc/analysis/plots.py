"""Render the analysis figures (matplotlib, modern technical style).

Style follows the house rules: no titles, hidden top/right spines, light
background, dark-grey ink, muted palette, error bars/bands everywhere, frameless
legend, constrained layout, 300 DPI PNG. Each figure is guarded so a missing
input table just skips that plot rather than failing the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .frameio import load_table  # noqa: E402

INK = "#333333"
ACCENT = "#3b6ea5"
POS = "#4a9c6d"
NEG = "#b5544a"
MUTED = "#9aa3ab"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e9e9e9", "grid.linewidth": 0.6,
        "font.size": 10, "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[plots] wrote {path}", file=sys.stderr)


def _try(name: str, fn, *args) -> None:
    try:
        fn(*args)
    except FileNotFoundError:
        print(f"[plots] skip {name}: input missing", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[plots] skip {name}: {type(e).__name__}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------


def plot_event_study(results: Path, out: Path) -> None:
    es = load_table(results, "event-study")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = es["t"].to_numpy()
    y = es["mean_excess"].to_numpy() * 100
    lo = es["lo"].to_numpy() * 100
    hi = es["hi"].to_numpy() * 100
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.fill_between(x, lo, hi, color=ACCENT, alpha=0.18, linewidth=0)
    ax.plot(x, y, color=ACCENT, lw=1.6)
    ax.set_xlabel("months relative to AI availability")
    ax.set_ylabel("excess source churn per developer (%)")
    ax.annotate("AI availability", xy=(0, ax.get_ylim()[1]), xytext=(2, ax.get_ylim()[1] * 0.9),
                color=MUTED, fontsize=9)
    ax.annotate("flat pre-period = no divergent trend", xy=(x.min(), 0),
                xytext=(x.min(), ax.get_ylim()[1] * 0.5), color=MUTED, fontsize=8)
    _save(fig, out / "fig-event-study.png")


def plot_dose_response(results: Path, out: Path) -> None:
    dose = load_table(results, "dose-response")
    summary = json.loads((results / "stats-summary.json").read_text())
    dr = summary.get("dose_response", {})
    x = dose["suitability_weighted"].to_numpy()
    y = dose["post_excess"].to_numpy() * 100  # log points ~ percent for small values

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.scatter(x, y, s=26, color=ACCENT, alpha=0.7, edgecolor="white", linewidth=0.5)
    if len(x) >= 2:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=INK, lw=1.4)
        # bootstrap band
        rng = np.random.default_rng(0)
        preds = []
        for _ in range(300):
            idx = rng.choice(len(x), len(x), replace=True)
            bb1, bb0 = np.polyfit(x[idx], y[idx], 1)
            preds.append(bb0 + bb1 * xs)
        p = np.percentile(preds, [2.5, 97.5], axis=0)
        ax.fill_between(xs, p[0], p[1], color=INK, alpha=0.10, linewidth=0)
    if "slope" in dr:
        ax.annotate(
            f"slope = {dr['slope'] * 100:.0f} pts/suitability\n"
            f"95% CI [{dr['slope_lo'] * 100:.0f}, {dr['slope_hi'] * 100:.0f}], "
            f"p = {dr['slope_p']:.1e}",
            xy=(0.03, 0.92), xycoords="axes fraction", va="top", fontsize=9, color=INK)
    ax.set_xlabel("language AI-suitability (churn-weighted)")
    ax.set_ylabel("post-AI excess churn (log points)")
    ax.annotate("niche langs\n(AI weak)", xy=(x.min(), y.min()), color=MUTED, fontsize=8)
    ax.annotate("mainstream\n(AI strong)", xy=(x.max() * 0.9, y.max()), color=MUTED, fontsize=8,
                ha="right")
    _save(fig, out / "fig-dose-response.png")


def plot_management_kpi(results: Path, out: Path) -> None:
    kpi = json.loads((results / "management-kpi.json").read_text())
    fig, ax = plt.subplots(figsize=(4.5, 4))
    vals = [100.0, kpi["index_baseline_100"]]
    ax.bar(["pre-AI\nbaseline", "post-AI"], vals, color=[MUTED, POS], width=0.6)
    ax.axhline(100, color=MUTED, lw=0.8, ls="--")
    ax.set_ylabel("gross SLOC added / dev-month (index, baseline = 100)")
    ax.annotate(f"{kpi['pct_change']:+.1f}%", xy=(1, kpi["index_baseline_100"]),
                xytext=(1, kpi["index_baseline_100"] + 2), ha="center", color=INK, fontweight="bold")
    ax.text(0.5, -0.22, "management KPI (raw volume) — see quality panels before\n"
            "calling it productivity", transform=ax.transAxes, ha="center", fontsize=7.5,
            color=MUTED)
    _save(fig, out / "fig-management-kpi.png")


def plot_propensity(results: Path, out: Path) -> None:
    p = load_table(results, "propensity-repo").dropna(subset=["p_ai"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    inferred = p[p["label_source"] == "inferred"]["p_ai"].to_numpy()
    sigpos = p[p["label_source"] == "signature-positive"]["p_ai"].to_numpy()
    bins = np.linspace(0, 1, 21)
    ax1.hist(inferred, bins=bins, color=ACCENT, alpha=0.75, label="inferred")
    ax1.hist(sigpos, bins=bins, color=POS, alpha=0.75, label="signature-positive")
    ax1.set_xlabel("P(repo uses AI)")
    ax1.set_ylabel("repositories")
    ax1.legend(loc="upper center")

    # forest of the top inferred repos with CIs
    top = p[p["label_source"] == "inferred"].nlargest(15, "p_ai").iloc[::-1]
    if not top.empty:
        yy = np.arange(len(top))
        xc = top["p_ai"].to_numpy()
        lo = top["p_ai_lo"].to_numpy()
        hi = top["p_ai_hi"].to_numpy()
        ax2.errorbar(xc, yy, xerr=[xc - lo, hi - xc], fmt="o", color=ACCENT,
                     ecolor=MUTED, elinewidth=1.2, capsize=2, ms=4)
        ax2.set_yticks(yy)
        labels = top["name"] if "name" in top.columns else top["repo_id"]
        ax2.set_yticklabels([str(s)[-22:] for s in labels], fontsize=7)
        ax2.set_xlabel("P(AI) with 95% bootstrap CI")
        ax2.set_xlim(0, 1.02)
    _save(fig, out / "fig-propensity.png")


def render_all(results: Path, out: Path) -> None:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    _try("event-study", plot_event_study, results, out)
    _try("dose-response", plot_dose_response, results, out)
    _try("management-kpi", plot_management_kpi, results, out)
    _try("propensity", plot_propensity, results, out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aisloc.analysis.plots", description=__doc__)
    p.add_argument("--results", default="data/results", type=Path)
    p.add_argument("--out", default="data/results", type=Path)
    a = p.parse_args(argv)
    render_all(a.results, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
