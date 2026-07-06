"""Render the analysis figures (matplotlib, modern technical style).

Style comes from the shared house stylesheets (modern.mplstyle / modern-dark.
mplstyle): hidden top/right spines, muted grid, frameless legend, constrained
layout, 300 DPI PNG. Every figure carries a title -- these are shared as
standalone files, not always alongside surrounding prose, so the title (not
just axis labels) has to carry the figure's meaning. Each figure is guarded so
a missing input table just skips that plot rather than failing the run.

Renders BOTH a light and a dark variant of every figure (the dark ones get a
"-dark" filename suffix) -- see render_all()/_apply_theme().
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .frameio import load_table  # noqa: E402

_SKILL_DIR = Path("/home/pfeifer/.claude/skills/matplotlib-style")


@dataclass(frozen=True)
class _Theme:
    style_path: Path
    suffix: str
    ink: str
    muted: str
    pre: str
    post: str
    pos: str
    neg: str
    seq: object  # sequential colormap, ordered/ranked data
    seq2: object  # second sequential colormap, for 2-tone gradients
    qual: object  # qualitative colormap with >10 distinguishable hues


_LIGHT = _Theme(
    style_path=_SKILL_DIR / "modern.mplstyle", suffix="",
    ink="#333333", muted="#9aa3ab",
    pre="#4C72B0", post="#DD8452", pos="#55A868", neg="#C44E52",
    seq=plt.cm.viridis, seq2=plt.cm.plasma, qual=plt.cm.tab20,
)
_DARK = _Theme(
    style_path=_SKILL_DIR / "modern-dark.mplstyle", suffix="-dark",
    ink="#e8e6f0", muted="#8f8aa8",
    pre="#40c4ff", post="#ffab40", pos="#69f0ae", neg="#ff6e6e",
    seq=plt.cm.plasma, seq2=plt.cm.magma, qual=plt.cm.tab20,
)

# Mutable "current theme" -- reassigned by _apply_theme() before each render
# pass; the plot functions below read these module-level names directly.
INK = _LIGHT.ink
MUTED = _LIGHT.muted
PRE = _LIGHT.pre
POST = _LIGHT.post
POS = _LIGHT.pos
NEG = _LIGHT.neg
SEQ = _LIGHT.seq
SEQ2 = _LIGHT.seq2
QUAL = _LIGHT.qual
_SUFFIX = _LIGHT.suffix


def _apply_theme(theme: _Theme) -> None:
    global INK, MUTED, PRE, POST, POS, NEG, SEQ, SEQ2, QUAL, _SUFFIX
    plt.style.use(str(theme.style_path))
    INK, MUTED = theme.ink, theme.muted
    PRE, POST, POS, NEG = theme.pre, theme.post, theme.pos, theme.neg
    SEQ, SEQ2, QUAL = theme.seq, theme.seq2, theme.qual
    _SUFFIX = theme.suffix


def _style() -> None:
    _apply_theme(_LIGHT)


def _save(fig: plt.Figure, path: Path) -> None:
    if _SUFFIX:
        path = path.with_name(f"{path.stem}{_SUFFIX}{path.suffix}")
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
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = es["t"].to_numpy()
    y = es["mean_excess"].to_numpy() * 100
    lo = es["lo"].to_numpy() * 100
    hi = es["hi"].to_numpy() * 100

    # Diverging line color keyed to t itself (cool = pre-AI, warm = post-AI),
    # centered exactly at the anchor -- encodes "before vs. after" through
    # color as well as position.
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(vmin=-max(abs(x.min()), abs(x.max())), vmax=max(abs(x.min()), abs(x.max())))
    lc = LineCollection(segments, cmap="coolwarm", norm=norm, linewidth=2.4)
    lc.set_array((x[:-1] + x[1:]) / 2)
    ax.add_collection(lc)
    ax.fill_between(x, lo, hi, color=INK, alpha=0.10, linewidth=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(min(lo.min(), y.min()) * 1.05, max(hi.max(), y.max()) * 1.1)
    ax.set_title("Excess Source Churn Around AI Availability")
    ax.set_xlabel("Months Relative to AI Availability")
    ax.set_ylabel("Excess Source Churn per Developer [%]")
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

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhline(0, color=MUTED, lw=0.8)
    sc = ax.scatter(x, y, s=32, c=x, cmap=SEQ, alpha=0.85, edgecolor="white", linewidth=0.5)
    fig.colorbar(sc, ax=ax, label="AI-Suitability (Same Scale as X-Axis)", pad=0.01)
    if len(x) >= 2:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=INK, lw=1.6)
        # bootstrap band
        rng = np.random.default_rng(0)
        preds = []
        for _ in range(300):
            idx = rng.choice(len(x), len(x), replace=True)
            bb1, bb0 = np.polyfit(x[idx], y[idx], 1)
            preds.append(bb0 + bb1 * xs)
        p = np.percentile(preds, [2.5, 97.5], axis=0)
        ax.fill_between(xs, p[0], p[1], color=INK, alpha=0.12, linewidth=0)
    if "slope" in dr:
        ax.annotate(
            f"slope = {dr['slope'] * 100:.0f} pts/suitability\n"
            f"95% CI [{dr['slope_lo'] * 100:.0f}, {dr['slope_hi'] * 100:.0f}], "
            f"p = {dr['slope_p']:.1e}",
            xy=(0.03, 0.92), xycoords="axes fraction", va="top", fontsize=9, color=INK)
    ax.set_title("Post-AI Excess Churn vs. Language AI-Suitability")
    ax.set_xlabel("Language AI-Suitability (Churn-Weighted)")
    ax.set_ylabel("Post-AI Excess Churn [Log Points]")
    ax.annotate("niche langs\n(AI weak)", xy=(x.min(), y.min()), color=MUTED, fontsize=8)
    ax.annotate("mainstream\n(AI strong)", xy=(x.max() * 0.9, y.max()), color=MUTED, fontsize=8,
                ha="right")
    _save(fig, out / "fig-dose-response.png")


def plot_management_kpi(results: Path, out: Path) -> None:
    kpi = json.loads((results / "management-kpi.json").read_text())
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    vals = [100.0, kpi["index_baseline_100"]]
    colors = SEQ2(np.linspace(0.25, 0.75, 2))
    ax.bar(["pre-AI\nbaseline", "post-AI"], vals, color=colors, width=0.6, edgecolor="none")
    ax.axhline(100, color=MUTED, lw=0.8, ls="--")
    ax.set_title("Gross Code Volume: Pre- vs. Post-AI")
    ax.set_ylabel("Gross SLOC Added per Dev-Month [Index, Baseline=100]")
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
    ax1.hist(inferred, bins=bins, color=PRE, alpha=0.85, label="inferred")
    ax1.hist(sigpos, bins=bins, color=POST, alpha=0.85, label="signature-positive")
    ax1.set_title("Distribution of P(AI)")
    ax1.set_xlabel("P(Repo Uses AI)")
    ax1.set_ylabel("Repositories")
    ax1.legend(loc="upper center")

    # forest of the top inferred repos with CIs, dots colored by rank (viridis
    # -- sequential, since these are already sorted by p_ai)
    top = p[p["label_source"] == "inferred"].nlargest(15, "p_ai").iloc[::-1]
    if not top.empty:
        yy = np.arange(len(top))
        xc = top["p_ai"].to_numpy()
        lo = top["p_ai_lo"].to_numpy()
        hi = top["p_ai_hi"].to_numpy()
        dot_colors = SEQ(np.linspace(0.15, 0.85, len(top)))
        ax2.errorbar(xc, yy, xerr=[xc - lo, hi - xc], fmt="none",
                     ecolor=MUTED, elinewidth=1.2, capsize=2, zorder=1)
        ax2.scatter(xc, yy, color=dot_colors, s=40, zorder=2, edgecolor="white", linewidth=0.6)
        ax2.set_yticks(yy)
        labels = top["name"] if "name" in top.columns else top["repo_id"]
        ax2.set_yticklabels([str(s)[-22:] for s in labels], fontsize=7)
        ax2.set_title("Top-Ranked Repositories by P(AI)")
        ax2.set_xlabel("P(AI) [95% Bootstrap CI]")
        ax2.set_xlim(0, 1.02)
    fig.suptitle("Estimated P(AI Use) Across Repositories", fontsize=13, color=INK)
    _save(fig, out / "fig-propensity.png")


def _plot_before_after_groups(
    groups: dict, pre_key: str, post_key: str, ylabel: str, filename: str, out: Path,
    title: str,
) -> None:
    labels = {"likely_ai": "likely AI-using\n(p_ai ≥ 0.7)",
              "unlikely_ai": "likely not\n(p_ai < 0.7)"}
    present = [g for g in ("likely_ai", "unlikely_ai") if groups.get(g, {}).get("n")]
    if not present:
        raise FileNotFoundError("no before/after groups with data")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(present))
    width = 0.32
    pre_vals = [groups[g][pre_key] for g in present]
    post_vals = [groups[g][post_key] for g in present]
    ax.bar(x - width / 2, pre_vals, width, color=PRE, label="pre-AI", edgecolor="none")
    ax.bar(x + width / 2, post_vals, width, color=POST, label="post-AI", edgecolor="none")
    for i, g in enumerate(present):
        pct = groups[g].get("pct_change")
        lo, hi = groups[g].get("pct_change_lo"), groups[g].get("pct_change_hi")
        if pct is None:
            continue
        ci = f"\n[{lo:+.0f}, {hi:+.0f}]" if lo is not None else ""
        top = max(pre_vals[i], post_vals[i])
        ax.annotate(f"{pct:+.0f}%{ci}", xy=(x[i], top), xytext=(x[i], top * 1.08),
                    ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{labels[g]}  (n={groups[g]['n']})" for g in present])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.15)
    ax.legend(loc="upper left")
    ax.text(
        0.99, -0.14,
        "descriptive, not counterfactual-corrected -- pair with dose-response + placebo",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED,
    )
    _save(fig, out / filename)


def plot_before_after(results: Path, out: Path) -> None:
    """Concrete pre- vs post-AI churn per developer-month, plain units, grouped
    by AI-likelihood (not counterfactual-corrected -- see before-after.csv
    module docstring in stats.py for what this does and doesn't show)."""
    summary = json.loads((results / "stats-summary.json").read_text())
    groups = summary.get("before_after_by_ai_group") or {}
    _plot_before_after_groups(
        groups, "mean_pre_churn_per_dev_month", "mean_post_churn_per_dev_month",
        "Source Churn per Developer-Month", "fig-before-after.png", out,
        title="Repo-Level Churn: Pre- vs. Post-AI, by P(AI) Group",
    )


def plot_dev_before_after(results: Path, out: Path) -> None:
    """Same contrast as plot_before_after, but at individual-developer
    granularity: each contributor's own churn, before vs after, grouped by
    their own PU-model p_ai. Immune to the repo-level team-size composition
    confound by construction (see stats.py's developer-level section)."""
    summary = json.loads((results / "stats-summary.json").read_text())
    groups = (summary.get("developer_level") or {}).get("before_after_by_p_ai_group") or {}
    _plot_before_after_groups(
        groups, "mean_pre_churn_per_month", "mean_post_churn_per_month",
        "Individual Source Churn per Month", "fig-dev-before-after.png", out,
        title="Individual Churn: Pre- vs. Post-AI, by P(AI) Group",
    )


def plot_dev_dose_response(results: Path, out: Path) -> None:
    """Individual-level analogue of plot_dose_response: each developer's own
    post-anchor excess churn against their own estimated P(AI use), not
    language suitability."""
    dose = load_table(results, "dev-dose-response")
    summary = json.loads((results / "stats-summary.json").read_text())
    dr = (summary.get("developer_level") or {}).get("dose_response_by_own_p_ai") or {}
    x = dose["p_ai"].to_numpy()
    y = dose["post_excess"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhline(0, color=MUTED, lw=0.8)
    sc = ax.scatter(x, y, s=20, c=x, cmap=SEQ, alpha=0.6, edgecolor="white", linewidth=0.3)
    fig.colorbar(sc, ax=ax, label="Developer's Own P(AI Use)", pad=0.01)
    if len(x) >= 2:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=INK, lw=1.6)
        rng = np.random.default_rng(0)
        preds = []
        for _ in range(300):
            idx = rng.choice(len(x), len(x), replace=True)
            bb1, bb0 = np.polyfit(x[idx], y[idx], 1)
            preds.append(bb0 + bb1 * xs)
        p = np.percentile(preds, [2.5, 97.5], axis=0)
        ax.fill_between(xs, p[0], p[1], color=INK, alpha=0.12, linewidth=0)
    if "slope" in dr:
        ax.annotate(
            f"slope = {dr['slope'] * 100:.0f} pts/p_ai\n"
            f"95% CI [{dr['slope_lo'] * 100:.0f}, {dr['slope_hi'] * 100:.0f}], "
            f"p = {dr['slope_p']:.1e}  (n={dr.get('n', len(x))})",
            xy=(0.03, 0.92), xycoords="axes fraction", va="top", fontsize=9, color=INK)
    ax.set_title("Post-AI Excess Churn vs. Developer's Own P(AI)")
    ax.set_xlabel("Developer's Own P(AI Use)")
    ax.set_ylabel("Individual Post-AI Excess Churn [Log Points]")
    _save(fig, out / "fig-dev-dose-response.png")


def plot_mentions_timeseries(results: Path, out: Path) -> None:
    """Dedicated, standalone "hype curve": raw bare-word AI-tool mention rate
    per month, deliberately never merged into the p_ai/dose-response figures
    above -- see stats.py's dedicated-analyses section docstring for why.
    Stacked by individual tool (qualitative palette) rather than one flat
    total line, so which tool actually drives the trend is visible."""
    m = load_table(results, "mentions-timeseries").sort_values("ym")
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(m))
    term_cols = [c for c in m.columns if c.startswith("mentions_")]
    # drop terms that never fire in this dataset -- an all-zero series would
    # just clutter the legend with invisible entries
    term_cols = [c for c in term_cols if m[c].sum() > 0]
    term_cols.sort(key=lambda c: m[c].sum(), reverse=True)
    shares = [m[c].to_numpy() / m["total_commits"].to_numpy() * 100 for c in term_cols]
    labels = [c.removeprefix("mentions_").replace("_", " ") for c in term_cols]
    # The stylesheet's prop_cycle only has ~10 qualitative colors; with up to
    # 11 tool terms that would silently recycle one (confirmed: "tabnine" and
    # "claude" came out an identical blue). QUAL (tab20) has enough distinct hues.
    colors = QUAL(np.linspace(0, 1, len(term_cols)))
    ax.stackplot(x, shares, labels=labels, colors=colors, alpha=0.9)
    ax.set_title("AI-Tool Mentions in Commit Messages Over Time")
    ax.set_ylabel("Commits Mentioning a Given AI Tool by Name [%]")
    step = max(1, len(m) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(m["ym"].to_numpy()[::step], rotation=60, ha="right", fontsize=8)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.text(0.99, 0.97, "raw bare-word mentions -- descriptive only, not a detection signal",
            transform=ax.transAxes, va="top", ha="right", fontsize=7.5, color=MUTED)
    _save(fig, out / "fig-mentions-timeseries.png")


def plot_style_changepoints(results: Path, out: Path) -> None:
    """Population histogram of detected per-developer commit-message-length
    breakpoints -- dedicated, standalone, kept out of the p_ai pipeline (see
    stats.py's dedicated-analyses section docstring). Only breakpoints that
    passed both the effect-size and permutation-significance filters are
    counted (style_changepoint_histogram in stats.py)."""
    h = load_table(results, "style-changepoint-histogram").sort_values("breakpoint_ym")
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(h))
    y = h["n_breakpoints"].to_numpy()

    # Gradient-colored line + fill (viridis, keyed to the value itself) --
    # a time series reads better as a line than as flat bars (see skill), and
    # the color gradient echoes the rising trend instead of a single flat hue.
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    lc = LineCollection(segments, cmap=SEQ, norm=norm, linewidth=2.4)
    lc.set_array((y[:-1] + y[1:]) / 2)
    ax.add_collection(lc)
    ax.fill_between(x, 0, y, color=SEQ(0.5), alpha=0.15, linewidth=0)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, y.max() * 1.1)
    ax.set_title("Detected Commit-Message-Length Breakpoints by Month")
    ax.set_ylabel("Developers with a Detected Message-Length Jump")
    step = max(1, len(h) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(h["breakpoint_ym"].to_numpy()[::step], rotation=60, ha="right", fontsize=8)
    ax.text(0.01, 0.97, "suggestive, never proof -- a habit change unrelated to AI\n"
            "produces the identical pattern (see stats.py docstring)",
            transform=ax.transAxes, va="top", fontsize=7.5, color=MUTED)
    _save(fig, out / "fig-style-changepoints.png")


def _render_pass(results: Path, out: Path) -> None:
    _try("event-study", plot_event_study, results, out)
    _try("dose-response", plot_dose_response, results, out)
    _try("management-kpi", plot_management_kpi, results, out)
    _try("propensity", plot_propensity, results, out)
    _try("before-after", plot_before_after, results, out)
    _try("dev-before-after", plot_dev_before_after, results, out)
    _try("dev-dose-response", plot_dev_dose_response, results, out)
    _try("mentions-timeseries", plot_mentions_timeseries, results, out)
    _try("style-changepoints", plot_style_changepoints, results, out)


def render_all(results: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for theme in (_LIGHT, _DARK):
        _apply_theme(theme)
        _render_pass(results, out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aisloc.analysis.plots", description=__doc__)
    p.add_argument("--results", default="data/results", type=Path)
    p.add_argument("--out", default="data/results", type=Path)
    a = p.parse_args(argv)
    render_all(a.results, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
