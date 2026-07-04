"""The statistical stack (concept.md sec. 2, 4, 6).

Consumes the tidy panels and produces, in data/results/:
  * event-study.csv       -- pooled excess-churn ATT(t) with CIs (Design A/B)
  * dose-response.csv      -- per-repo post excess vs. language AI-suitability
  * backbone.csv           -- OLS with cluster-robust SE; post x suitability = key
  * management-kpi.json     -- raw-volume "productivity" headline (sec. 4.2)
  * placebo.json           -- fake-anchor check; excess must vanish
  * propensity-repo.csv     -- calibrated P(AI) per repo (PU / Elkan-Noto, Design E)
  * propensity-author.csv   -- calibrated P(AI) per developer
  * stats-summary.json      -- headline numbers

Everything is numpy/scipy/pandas only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .frameio import load_table, save_table
from .inclusion import InclusionRule
from .model import logistic_fit, logistic_predict, ols_cluster, zscore

MIN_PRE = 12          # months of pre-AI history needed to fit a counterfactual
POST_WINDOW = 24      # months after anchor used for the dose-response summary
ES_LO, ES_HI = -24, 30
PLACEBO_ANCHOR = "2020-06"


def _ord(ym: str) -> int:
    y, m = ym.split("-")
    return int(y) * 12 + (int(m) - 1)


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


# ---------------------------------------------------------------------------
# data prep
# ---------------------------------------------------------------------------


def _repo_month(panels: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rm = load_table(panels, "repo-month")
    am = load_table(panels, "author-month")
    rmt = rm.groupby(["repo_id", "ym", "t"], as_index=False).agg(
        churn=("churn", "sum"), ins=("ins", "sum"), commits=("commits", "sum")
    )
    devs = (
        am.groupby(["repo_id", "ym"], as_index=False)["dev"]
        .nunique().rename(columns={"dev": "devs"})
    )
    rmt = rmt.merge(devs, on=["repo_id", "ym"], how="left")
    rmt["devs"] = rmt["devs"].fillna(1).clip(lower=1)
    rmt["cpd"] = rmt["churn"] / rmt["devs"]
    rmt["month"] = rmt["ym"].str.slice(5, 7).astype(int)
    return rmt, am


# ---------------------------------------------------------------------------
# counterfactual excess (Design A)
# ---------------------------------------------------------------------------


def _counterfactual(rmt: pd.DataFrame, anchor_col: str = "t") -> pd.DataFrame:
    """Fit log(cpd) ~ t + month-of-year on each repo's pre-period, project it,
    return residual excess per (repo, t)."""
    rows = []
    for rid, g in rmt.groupby("repo_id"):
        g = g.sort_values(anchor_col)
        pre = g[g[anchor_col] < 0]
        if len(pre) < MIN_PRE:
            continue
        Xpre = _design(pre, anchor_col)
        ypre = np.log1p(pre["cpd"].to_numpy())
        beta = np.linalg.pinv(Xpre.T @ Xpre) @ (Xpre.T @ ypre)
        Xall = _design(g, anchor_col)
        yhat = Xall @ beta
        resid = np.log1p(g["cpd"].to_numpy()) - yhat
        rows.append(pd.DataFrame({
            "repo_id": rid, "t": g[anchor_col].to_numpy(),
            "resid": resid, "pct": np.expm1(resid),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["repo_id", "t", "resid", "pct"]
    )


def _design(g: pd.DataFrame, tcol: str) -> np.ndarray:
    t = g[tcol].to_numpy(dtype=float)
    month = g["month"].to_numpy()
    dummies = np.zeros((len(g), 11))
    for i, m in enumerate(month):
        if 1 <= m <= 11:
            dummies[i, m - 1] = 1.0
    return np.column_stack([np.ones(len(g)), t, dummies])


def _event_study(excess: pd.DataFrame) -> pd.DataFrame:
    # Aggregate in log space (residuals are ~symmetric, mean-zero under the null);
    # averaging expm1() directly would inject a positive Jensen bias so that even
    # pure noise looks like "excess". Convert to percent only after averaging.
    e = excess[(excess["t"] >= ES_LO) & (excess["t"] <= ES_HI)]
    out = []
    for t, g in e.groupby("t"):
        vals = g["resid"].to_numpy()
        n = len(vals)
        mean_r = float(np.mean(vals))
        sem_r = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        out.append({"t": int(t), "mean_excess": float(np.expm1(mean_r)), "sem": sem_r, "n": n,
                    "lo": float(np.expm1(mean_r - 1.96 * sem_r)),
                    "hi": float(np.expm1(mean_r + 1.96 * sem_r))})
    return pd.DataFrame(out).sort_values("t")


# ---------------------------------------------------------------------------
# dose-response + backbone
# ---------------------------------------------------------------------------


def _dose_response(excess: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    post = excess[(excess["t"] >= 0) & (excess["t"] <= POST_WINDOW)]
    # mean residual = mean log excess per repo (unbiased); slope is in log points
    per_repo = post.groupby("repo_id", as_index=False)["resid"].mean().rename(
        columns={"resid": "post_excess"})
    per_repo = per_repo.merge(
        meta[["repo_id", "suitability_weighted", "primary_language"]], on="repo_id", how="left"
    )
    per_repo = per_repo.dropna(subset=["suitability_weighted"])
    stat: dict[str, float | int] = {"n": int(len(per_repo))}
    if len(per_repo) >= 5:
        X = np.column_stack([np.ones(len(per_repo)), per_repo["suitability_weighted"].to_numpy()])
        y = per_repo["post_excess"].to_numpy()
        res = ols_cluster(X, y, ["const", "suitability"])
        s = res.summary()[1]
        stat.update({"slope": s["coef"], "slope_lo": s["ci_lo"], "slope_hi": s["ci_hi"],
                     "slope_p": s["p"]})
    return per_repo.sort_values("suitability_weighted"), stat


def _backbone(rmt: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = rmt.merge(meta[["repo_id", "suitability_weighted", "included"]], on="repo_id", how="left")
    df = df[_as_bool(df["included"])].dropna(subset=["suitability_weighted"])
    if len(df) < 30:
        return pd.DataFrame()
    post = (df["t"] >= 0).to_numpy(dtype=float)
    suit_c = df["suitability_weighted"].to_numpy() - df["suitability_weighted"].mean()
    X = np.column_stack([np.ones(len(df)), post, suit_c, post * suit_c, df["t"].to_numpy()])
    y = np.log1p(df["cpd"].to_numpy())
    groups = df["repo_id"].to_numpy()
    res = ols_cluster(X, y, ["const", "post", "suitability", "post_x_suitability", "t"], groups)
    return pd.DataFrame(res.summary())


def _management_kpi(rmt: pd.DataFrame, meta: pd.DataFrame) -> dict:
    df = rmt.merge(meta[["repo_id", "included"]], on="repo_id", how="left")
    df = df[_as_bool(df["included"])]
    df["added_per_dev"] = df["ins"] / df["devs"]
    pre = df.loc[df["t"] < 0, "added_per_dev"].mean()
    post = df.loc[df["t"] >= 0, "added_per_dev"].mean()
    index = 100.0 * post / pre if pre else float("nan")
    return {"pre_added_per_dev_month": float(pre), "post_added_per_dev_month": float(post),
            "index_baseline_100": float(index), "pct_change": float(index - 100.0),
            "caveat": "raw volume; see durable-output/quality panels before calling it productivity"}


def _placebo(rmt: pd.DataFrame) -> dict:
    anchor = _ord(rmt["ym"].min()[:4] + "-01")  # unused; compute placebo t
    plac = _ord(PLACEBO_ANCHOR)
    df = rmt.copy()
    df["t"] = df["ym"].map(_ord) - plac
    # only use history before the real anchor so the real effect can't leak in
    df = df[df["ym"] < InclusionRule().anchor]
    ex = _counterfactual(df, "t")
    post = ex[(ex["t"] >= 0) & (ex["t"] <= 18)]["resid"]
    return {"anchor": PLACEBO_ANCHOR, "n": int(len(post)),
            "mean_excess": float(np.expm1(post.mean())) if len(post) else float("nan")}


# ---------------------------------------------------------------------------
# PU propensity (Design E)
# ---------------------------------------------------------------------------


def _repo_features(rmt: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in rmt.groupby("repo_id"):
        pre, post = g[g["t"] < 0], g[g["t"] >= 0]
        if len(pre) < 3 or len(post) < 3:
            continue
        cpd_pre, cpd_post = pre["cpd"].mean(), post["cpd"].mean()
        size_pre = (pre["churn"].sum() / max(1, pre["commits"].sum()))
        size_post = (post["churn"].sum() / max(1, post["commits"].sum()))
        burst = post["cpd"].std() / cpd_post if cpd_post else 0.0
        slope = np.polyfit(post["t"], post["cpd"], 1)[0] / (cpd_post + 1) if len(post) > 2 else 0.0
        rows.append({
            "repo_id": rid,
            "f_churn_ratio": np.log((cpd_post + 1) / (cpd_pre + 1)),
            "f_size_ratio": np.log((size_post + 1) / (size_pre + 1)),
            "f_burst": float(burst) if np.isfinite(burst) else 0.0,
            "f_growth": float(slope) if np.isfinite(slope) else 0.0,
        })
    feats = pd.DataFrame(rows)
    return feats.merge(
        meta[["repo_id", "suitability_weighted", "has_signature", "name", "url"]],
        on="repo_id", how="left",
    ).rename(columns={"suitability_weighted": "f_suit"})


def _pu_propensity(feats: pd.DataFrame, label_col: str, id_cols: list[str],
                   n_boot: int = 200, seed: int = 7) -> pd.DataFrame:
    fcols = [c for c in feats.columns if c.startswith("f_")]
    df = feats.dropna(subset=fcols).copy()
    y = _as_bool(df[label_col]).to_numpy(dtype=float)
    base = df[id_cols].reset_index(drop=True)
    if y.sum() < 3 or len(df) < 10:
        # too few positives to learn; fall back to label as probability
        base["p_ai"] = np.where(y > 0, 0.95, np.nan)
        base["p_ai_lo"] = np.nan
        base["p_ai_hi"] = np.nan
        base["label_source"] = np.where(y > 0, "signature", "insufficient-positives")
        return base

    Xz, mu, sd = zscore(df[fcols].to_numpy())
    X = np.column_stack([np.ones(len(df)), Xz])

    def fit_predict(mask: np.ndarray) -> np.ndarray:
        w = logistic_fit(X[mask], y[mask])
        p = logistic_predict(X, w)
        c = p[y > 0].mean()  # Elkan-Noto: P(labeled | positive)
        return np.clip(p / max(c, 1e-6), 0, 1)

    point = fit_predict(np.ones(len(df), dtype=bool))
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(df)))
    idx = np.arange(len(df))
    for b in range(n_boot):
        sample = rng.choice(idx, size=len(df), replace=True)
        mask = np.zeros(len(df), dtype=bool)
        mask[sample] = True
        if y[mask].sum() < 2:
            boot[b] = point
            continue
        boot[b] = fit_predict(mask)

    base["p_ai"] = np.round(point, 4)
    base["p_ai_lo"] = np.round(np.percentile(boot, 2.5, axis=0), 4)
    base["p_ai_hi"] = np.round(np.percentile(boot, 97.5, axis=0), 4)
    base["label_source"] = np.where(y > 0, "signature-positive", "inferred")
    # signature-positive entities are AI by construction
    base.loc[y > 0, "p_ai"] = np.maximum(base.loc[y > 0, "p_ai"], 0.99)
    return base.sort_values("p_ai", ascending=False)


def _author_features(am: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rid, dev), g in am.groupby(["repo_id", "dev"]):
        pre, post = g[g["t"] < 0], g[g["t"] >= 0]
        if len(post) < 2:
            continue
        cpd_pre = pre["churn"].mean() if len(pre) else g["churn"].mean()
        cpd_post = post["churn"].mean()
        rows.append({
            "repo_id": rid, "dev": dev,
            "f_churn_ratio": np.log((cpd_post + 1) / (cpd_pre + 1)),
            "f_commits": np.log1p(post["commits"].sum()),
            "f_active_post": len(post),
            "dev_has_sig": bool(_as_bool(g["dev_has_sig"]).any()),
        })
    feats = pd.DataFrame(rows)
    if feats.empty:
        return feats
    return feats.merge(meta[["repo_id", "suitability_weighted"]], on="repo_id", how="left").rename(
        columns={"suitability_weighted": "f_suit"}
    )


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run(panels: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    meta = load_table(panels, "repo-meta")
    meta["included"] = _as_bool(meta["included"])
    meta["has_signature"] = _as_bool(meta["has_signature"])
    rmt, am = _repo_month(panels)

    inc_ids = set(meta.loc[meta["included"], "repo_id"])
    rmt_inc = rmt[rmt["repo_id"].isin(inc_ids)]

    excess = _counterfactual(rmt_inc)
    es = _event_study(excess)
    save_table(es, out, "event-study")

    dose, dose_stat = _dose_response(excess, meta)
    save_table(dose, out, "dose-response")

    backbone = _backbone(rmt, meta)
    if not backbone.empty:
        save_table(backbone, out, "backbone")

    kpi = _management_kpi(rmt, meta)
    (out / "management-kpi.json").write_text(json.dumps(kpi, indent=2))
    placebo = _placebo(rmt)
    (out / "placebo.json").write_text(json.dumps(placebo, indent=2))

    rfeats = _repo_features(rmt, meta)
    prop_repo = _pu_propensity(rfeats, "has_signature", ["repo_id", "name", "url"])
    save_table(prop_repo, out, "propensity-repo")

    afeats = _author_features(am, meta)
    if not afeats.empty:
        prop_auth = _pu_propensity(afeats, "dev_has_sig", ["repo_id", "dev"], n_boot=100)
        save_table(prop_auth, out, "propensity-author")

    post_es = es[(es["t"] >= 0) & (es["t"] <= POST_WINDOW)]["mean_excess"]
    summary = {
        "repos_total": int(len(meta)),
        "repos_included": int(meta["included"].sum()),
        "repos_with_signature": int(meta["has_signature"].sum()),
        "mean_post_excess": float(post_es.mean()) if len(post_es) else None,
        "dose_response": dose_stat,
        "management_kpi": kpi,
        "placebo_mean_excess": placebo["mean_excess"],
        "backbone_post_x_suitability": (
            next((r for r in backbone.to_dict("records")
                  if r["term"] == "post_x_suitability"), None)
            if not backbone.empty else None
        ),
    }
    (out / "stats-summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aisloc.analysis.stats", description=__doc__)
    p.add_argument("--panels", default="data/panels", type=Path)
    p.add_argument("--out", default="data/results", type=Path)
    a = p.parse_args(argv)
    s = run(a.panels, a.out)
    dr = s.get("dose_response", {})

    def f(x: object) -> str:
        return f"{x:+.3f}" if isinstance(x, (int, float)) else "n/a"

    print(
        f"[stats] included={s['repos_included']} sig={s['repos_with_signature']} | "
        f"mean post-excess={f(s['mean_post_excess'])} | "
        f"dose-response slope={f(dr.get('slope'))} (p={dr.get('slope_p', float('nan')):.1e}) | "
        f"placebo={f(s['placebo_mean_excess'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
