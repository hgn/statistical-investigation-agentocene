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
from datetime import datetime, timezone
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
# Winsorize log-residuals before expm1 (concept.md sec. 5.4): a quadratic trend
# fit on a short pre-period and extrapolated far into the post window can blow
# up numerically (float overflow) or dominate a mean with a single absurd
# outlier. +-5 log points is e^5-1 =~ +14700%/-99.3%, generous for any genuine
# change but a hard stop against extrapolation blow-up.
RESID_CLIP = 5.0


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
        churn=("churn", "sum"), ins=("ins", "sum"), dele=("del", "sum"), commits=("commits", "sum")
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
        resid = np.clip(np.log1p(g["cpd"].to_numpy()) - yhat, -RESID_CLIP, RESID_CLIP)
        rows.append(pd.DataFrame({
            "repo_id": rid, "t": g[anchor_col].to_numpy(),
            "resid": resid, "pct": np.expm1(resid),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["repo_id", "t", "resid", "pct"]
    )


def _design(g: pd.DataFrame, tcol: str) -> np.ndarray:
    """Linear trend plus a single annual Fourier pair for seasonality (2 params
    instead of 11 month dummies -- a harmless simplification kept from an
    earlier attempt at this function).

    A quadratic trend term was tried here (and in ``_dev_design``) to absorb
    lifecycle curvature, but naive polynomial extrapolation over the 24-30
    month post window turned out to be numerically unstable (small pre-period
    curvature amplifies wildly once squared and projected that far out) and
    made BOTH the repo- and developer-level placebo checks *worse*, not
    better (see git history / concept.md sec. 10). Reverted to linear. The
    lifecycle/lifecycle-adjacent confound this was meant to fix is instead
    handled where it can be identified safely: as an explicit age covariate in
    the pooled cross-repo ``_backbone`` regression, not via per-entity
    extrapolation.
    """
    t = g[tcol].to_numpy(dtype=float)
    month = g["month"].to_numpy(dtype=float)
    angle = 2 * np.pi * month / 12
    return np.column_stack([np.ones(len(g)), t, np.sin(angle), np.cos(angle)])


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
    if not out:
        return pd.DataFrame(columns=["t", "mean_excess", "sem", "n", "lo", "hi"])
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
    """Pooled cross-repo regression. Unlike the per-repo counterfactual, repo
    age (months since THAT repo's own first observed activity) is not merely an
    affine function of calendar t here: different repos start at different
    calendar times, so age varies independently of t across the pooled sample.
    Repos further along in their lifecycle when they hit the anchor (mature,
    stabilising) vs repos still early (high growth) is exactly the confound the
    placebo test caught at repo level -- age (+ age^2, to allow deceleration)
    is included so post x suitability isn't picking up "which repos happened
    to be older" instead of a genuine AI dose-response."""
    df = rmt.merge(meta[["repo_id", "suitability_weighted", "included"]], on="repo_id", how="left")
    df = df[_as_bool(df["included"])].dropna(subset=["suitability_weighted"])
    if len(df) < 30:
        return pd.DataFrame()
    df = df.copy()
    df["age"] = df["t"] - df.groupby("repo_id")["t"].transform("min")
    age_c = df["age"].to_numpy(dtype=float) - df["age"].mean()
    post = (df["t"] >= 0).to_numpy(dtype=float)
    suit_c = df["suitability_weighted"].to_numpy() - df["suitability_weighted"].mean()
    X = np.column_stack([
        np.ones(len(df)), post, suit_c, post * suit_c, df["t"].to_numpy(), age_c, age_c**2,
    ])
    y = np.log1p(df["cpd"].to_numpy())
    groups = df["repo_id"].to_numpy()
    res = ols_cluster(
        X, y, ["const", "post", "suitability", "post_x_suitability", "t", "age", "age_sq"], groups
    )
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
    if not rows:
        return pd.DataFrame(columns=["repo_id", "f_churn_ratio", "f_size_ratio", "f_burst",
                                      "f_growth", "suitability_weighted", "has_signature",
                                      "name", "url"])
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
# concrete before/after (plain units, not log residuals)
# ---------------------------------------------------------------------------
#
# The event-study/dose-response results above answer "is there an effect,
# robust to secular trends and noise" in log-residual terms, which is the
# right construct for *identification* but not the most legible one for a
# concrete "how much did code growth actually change" question. This section
# answers that directly, per repo and pooled by AI-likelihood group, in plain
# units (lines per developer-month) -- descriptive, not causal on its own: no
# counterfactual/placebo correction here, so read it alongside, not instead of,
# the dose-response and placebo results.

AI_GROUP_THRESHOLD = 0.7  # p_ai >= this -> "likely AI" group in the pooled comparison


def _before_after_repo(rmt: pd.DataFrame, meta: pd.DataFrame, prop_repo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in rmt.groupby("repo_id"):
        pre, post = g[g["t"] < 0], g[g["t"] >= 0]
        if len(pre) < 3 or len(post) < 3:
            continue
        devs_pre = pre["devs"].replace(0, 1)
        devs_post = post["devs"].replace(0, 1)
        pre_ins_dev = float((pre["ins"] / devs_pre).mean())
        post_ins_dev = float((post["ins"] / devs_post).mean())
        pre_churn_dev = float((pre["churn"] / devs_pre).mean())
        post_churn_dev = float((post["churn"] / devs_post).mean())
        pre_net_dev = float(((pre["ins"] - pre["dele"]) / devs_pre).mean())
        post_net_dev = float(((post["ins"] - post["dele"]) / devs_post).mean())
        rows.append({
            "repo_id": rid,
            "pre_ins_per_dev_month": round(pre_ins_dev, 1),
            "post_ins_per_dev_month": round(post_ins_dev, 1),
            "pct_change_ins": _pct_change(pre_ins_dev, post_ins_dev),
            "pre_churn_per_dev_month": round(pre_churn_dev, 1),
            "post_churn_per_dev_month": round(post_churn_dev, 1),
            "pct_change_churn": _pct_change(pre_churn_dev, post_churn_dev),
            "pre_net_per_dev_month": round(pre_net_dev, 1),
            "post_net_per_dev_month": round(post_net_dev, 1),
            "pct_change_net": _pct_change(pre_net_dev, post_net_dev),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.merge(
        meta[["repo_id", "name", "url", "primary_language"]], on="repo_id", how="left"
    )
    if not prop_repo.empty:
        df = df.merge(prop_repo[["repo_id", "p_ai", "label_source"]], on="repo_id", how="left")
    return df.sort_values("p_ai", ascending=False, na_position="last")


def _pct_change(pre: float, post: float) -> float | None:
    if pre == 0:
        return None
    return round(100.0 * (post - pre) / abs(pre), 1)


def _before_after_groups(
    before_after: pd.DataFrame,
    pre_col: str = "pre_churn_per_dev_month",
    post_col: str = "post_churn_per_dev_month",
) -> dict:
    """Pool a before/after table into an AI-likely vs AI-unlikely contrast: the
    concrete "what changed, before to after" number per group, with a bootstrap
    CI on the group mean. Shared by the repo-level and developer-level tables
    (see ``_before_after_dev`` below), which differ only in column names."""
    if before_after.empty or "p_ai" not in before_after.columns:
        return {}
    df = before_after.dropna(subset=["p_ai"])
    groups = {
        "likely_ai": df[df["p_ai"] >= AI_GROUP_THRESHOLD],
        "unlikely_ai": df[df["p_ai"] < AI_GROUP_THRESHOLD],
    }
    out: dict[str, dict] = {}
    for gname, gdf in groups.items():
        if gdf.empty:
            out[gname] = {"n": 0}
            continue
        out[gname] = {
            "n": int(len(gdf)),
            f"mean_{pre_col}": round(float(gdf[pre_col].mean()), 1),
            f"mean_{post_col}": round(float(gdf[post_col].mean()), 1),
            **_bootstrap_pct_change(gdf[pre_col].to_numpy(), gdf[post_col].to_numpy()),
        }
    return out


def _bootstrap_pct_change(
    pre: np.ndarray, post: np.ndarray, n_boot: int = 2000, seed: int = 11
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pre)
    if n == 0 or pre.mean() == 0:
        return {"pct_change": None, "pct_change_lo": None, "pct_change_hi": None}
    idx = np.arange(n)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        pm = pre[s].mean()
        samples[b] = 100.0 * (post[s].mean() - pm) / pm if pm != 0 else np.nan
    samples = samples[np.isfinite(samples)]
    point = 100.0 * (post.mean() - pre.mean()) / pre.mean()
    return {
        "pct_change": round(float(point), 1),
        "pct_change_lo": round(float(np.percentile(samples, 2.5)), 1) if len(samples) else None,
        "pct_change_hi": round(float(np.percentile(samples, 97.5)), 1) if len(samples) else None,
    }


# ---------------------------------------------------------------------------
# developer-level analysis (individual before/after, dosed by own p_ai)
# ---------------------------------------------------------------------------
#
# AI adoption is a developer-level choice, not a repo-level one: within the
# same project some contributors use it heavily, others not at all. Pooling to
# the repo mean dilutes exactly the signal we want, and (as the placebo check
# above revealed) is confounded by team-size composition changes over a
# project's lifetime -- a repo gaining more casual contributors mechanically
# lowers "churn per developer" with nothing to do with AI. Tracking the SAME
# person's own trajectory over time sidesteps that: team growth elsewhere
# doesn't move an individual's own churn.
#
# The dose here is the person's own PU-model p_ai (propensity-author), not
# language suitability -- directly testing "do developers who probably use AI
# show a bigger change in their own output". Same counterfactual/placebo
# discipline as the repo-level analysis applies: an individual can simply get
# more efficient with tenure on a project regardless of AI, so this still
# needs its own placebo check, not just a raw before/after.

DEV_MIN_PRE = 4    # active pre-months required (back to the original floor;
                   # see _dev_design docstring for why no t^2 term needs the
                   # extra degree of freedom anymore)
DEV_MIN_POST = 3   # active post-months required
DEV_POST_WINDOW = 24


def _dev_design(g: pd.DataFrame) -> np.ndarray:
    # No seasonal terms here (unlike the repo-level _design): a typical
    # contributor has far too few active months to identify them without
    # overfitting. Linear trend only -- a t^2 term was tried (for the same
    # lifecycle-curvature reasoning as _design) but blew up the placebo check
    # via unstable extrapolation over the post window; see _design's docstring.
    t = g["t"].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(g)), t])


def _dev_counterfactual(am: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rid, dev), g in am.groupby(["repo_id", "dev"]):
        g = g.sort_values("t")
        pre = g[g["t"] < 0]
        post = g[g["t"] >= 0]
        if len(pre) < DEV_MIN_PRE or len(post) < DEV_MIN_POST:
            continue
        Xpre = _dev_design(pre)
        ypre = np.log1p(pre["churn"].to_numpy())
        beta = np.linalg.pinv(Xpre.T @ Xpre) @ (Xpre.T @ ypre)
        Xall = _dev_design(g)
        resid = np.clip(np.log1p(g["churn"].to_numpy()) - Xall @ beta, -RESID_CLIP, RESID_CLIP)
        rows.append(pd.DataFrame({
            "repo_id": rid, "dev": dev, "t": g["t"].to_numpy(),
            "resid": resid, "pct": np.expm1(resid),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["repo_id", "dev", "t", "resid", "pct"]
    )


def _dev_placebo(am: pd.DataFrame) -> dict:
    plac = _ord(PLACEBO_ANCHOR)
    df = am.copy()
    df["t"] = df["ym"].map(_ord) - plac
    df = df[df["ym"] < InclusionRule().anchor]  # pre-real-anchor only
    ex = _dev_counterfactual(df)
    post = ex[(ex["t"] >= 0) & (ex["t"] <= 18)]["resid"]
    return {"anchor": PLACEBO_ANCHOR, "n": int(len(post)),
            "mean_excess": float(np.expm1(post.mean())) if len(post) else float("nan")}


def _dev_dose_response(excess: pd.DataFrame, prop_auth: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Regress each developer's own post-anchor excess churn on their own
    PU-model p_ai -- the direct individual-level analogue of the repo-level
    language-suitability dose-response."""
    post = excess[(excess["t"] >= 0) & (excess["t"] <= DEV_POST_WINDOW)]
    per_dev = post.groupby(["repo_id", "dev"], as_index=False)["resid"].mean().rename(
        columns={"resid": "post_excess"})
    if prop_auth.empty:
        return per_dev, {"n": 0}
    per_dev = per_dev.merge(
        prop_auth[["repo_id", "dev", "p_ai", "label_source"]], on=["repo_id", "dev"], how="left"
    )
    per_dev = per_dev.dropna(subset=["p_ai"])
    stat: dict[str, float | int] = {"n": int(len(per_dev))}
    if len(per_dev) >= 5:
        X = np.column_stack([np.ones(len(per_dev)), per_dev["p_ai"].to_numpy()])
        y = per_dev["post_excess"].to_numpy()
        res = ols_cluster(X, y, ["const", "p_ai"], groups=per_dev["repo_id"].to_numpy())
        s = res.summary()[1]
        stat.update({"slope": s["coef"], "slope_lo": s["ci_lo"], "slope_hi": s["ci_hi"],
                     "slope_p": s["p"]})
    return per_dev.sort_values("p_ai", ascending=False), stat


def _before_after_dev(am: pd.DataFrame, prop_auth: pd.DataFrame) -> pd.DataFrame:
    """Concrete, plain-units before/after per developer (own churn, no team-size
    division -- immune to the composition confound by construction)."""
    rows = []
    for (rid, dev), g in am.groupby(["repo_id", "dev"]):
        pre, post = g[g["t"] < 0], g[g["t"] >= 0]
        if len(pre) < DEV_MIN_PRE or len(post) < DEV_MIN_POST:
            continue
        pre_churn = float(pre["churn"].mean())
        post_churn = float(post["churn"].mean())
        rows.append({
            "repo_id": rid, "dev": dev,
            "pre_churn_per_month": round(pre_churn, 1),
            "post_churn_per_month": round(post_churn, 1),
            "pct_change_churn": _pct_change(pre_churn, post_churn),
        })
    df = pd.DataFrame(rows)
    if df.empty or prop_auth.empty:
        return df
    df = df.merge(
        prop_auth[["repo_id", "dev", "p_ai", "label_source"]], on=["repo_id", "dev"], how="left"
    )
    return df.sort_values("p_ai", ascending=False, na_position="last")


# ---------------------------------------------------------------------------
# individual heterogeneity: AI helps some developers a lot and others not at
# all, so a population-mean slope can look flat/negative while a real subset
# of individuals shows a large, genuine jump. Look at the distribution and
# the tail, not just the mean.
# ---------------------------------------------------------------------------

BIG_JUMP_THRESHOLD = 0.5  # own post-anchor excess >= +50% vs personal counterfactual


def _not_clipped(post_excess: pd.Series) -> pd.Series:
    """A dev whose mean post-anchor residual sits exactly at +-RESID_CLIP had
    every post-period month individually clipped -- a degenerate personal fit
    (near-empty or wildly noisy pre-period), not a genuine observation. These
    must not populate a "top gainers" leaderboard or dispersion stats: every
    such case would tie at the exact same clipped value, which is an artifact
    of the winsorization bound, not a real standout individual."""
    return post_excess.abs() < (RESID_CLIP - 1e-6)


def _dev_heterogeneity(dev_dose: pd.DataFrame) -> dict:
    """Per AI-likelihood group: share of individuals with a large personal
    jump, and dispersion of outcomes (std, p90) -- not just the mean, which is
    exactly what would hide a heterogeneous, individual-specific effect."""
    if dev_dose.empty or "p_ai" not in dev_dose.columns:
        return {}
    df = dev_dose.dropna(subset=["p_ai"]).copy()
    clip_rate = round(float((~_not_clipped(df["post_excess"])).mean()), 4)
    df = df[_not_clipped(df["post_excess"])]
    df["pct"] = np.expm1(df["post_excess"])
    df["big_jump"] = df["pct"] >= BIG_JUMP_THRESHOLD
    groups = {
        "likely_ai": df[df["p_ai"] >= AI_GROUP_THRESHOLD],
        "unlikely_ai": df[df["p_ai"] < AI_GROUP_THRESHOLD],
    }
    out: dict[str, dict] = {"clipped_share_excluded": clip_rate}
    for gname, gdf in groups.items():
        if gdf.empty:
            out[gname] = {"n": 0}
            continue
        out[gname] = {
            "n": int(len(gdf)),
            "share_big_jump_ge50pct": round(float(gdf["big_jump"].mean()), 3),
            "std_pct_excess": round(float(gdf["pct"].std()), 3),
            "p90_pct_excess": round(float(gdf["pct"].quantile(0.9)), 3),
            "median_pct_excess": round(float(gdf["pct"].median()), 3),
        }
    return out


def _dev_top_gainers(dev_dose: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """The standout individuals: largest personal post-anchor excess, for a
    concrete look at who these "very individual" increases belong to, alongside
    their own p_ai. Descriptive only -- a leaderboard, not a claim. Excludes
    clipped (degenerate-fit) rows, which would otherwise tie at the winsorized
    bound and masquerade as the biggest "winners"."""
    if dev_dose.empty:
        return dev_dose
    df = dev_dose.dropna(subset=["p_ai"]).copy()
    df = df[_not_clipped(df["post_excess"])]
    df["pct_excess"] = np.expm1(df["post_excess"]).round(3)
    return df.nlargest(n, "pct_excess")[["repo_id", "dev", "p_ai", "label_source", "pct_excess"]]


# ---------------------------------------------------------------------------
# productivity-tier stratification: the population mean can hide a tier-
# specific effect. An already-hyperactive core maintainer may show little room
# to grow (already near their personal ceiling, or simply not needing the
# help); a "second-row" contributor with more headroom might show a much
# larger relative AI effect that a pooled mean washes out (this is exactly the
# pattern the mean/median divergence in the before/after tables hinted at:
# a handful of high-baseline individuals can dominate an unweighted average).
# Two tier definitions are computed side by side rather than picking one, so a
# real finding isn't an artifact of the specific binning choice:
#   * "relative": terciles of pre-anchor baseline churn WITHIN each language
#     (fair across ecosystems with very different typical commit sizes);
#   * "absolute": fixed global churn/month bands, independent of language.
# ---------------------------------------------------------------------------

ABSOLUTE_TIER_BINS = [-np.inf, 200, 2000, np.inf]  # churn/month: low, mid, high
ABSOLUTE_TIER_LABELS = ["low", "mid", "high"]


def _dev_baseline(am: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per (repo_id, dev): pre-anchor baseline churn/month and the repo's
    primary language, the two inputs the tier assignment needs."""
    pre = am[am["t"] < 0]
    baseline = pre.groupby(["repo_id", "dev"], as_index=False)["churn"].mean().rename(
        columns={"churn": "baseline_churn"}
    )
    return baseline.merge(meta[["repo_id", "primary_language"]], on="repo_id", how="left")


def _assign_tiers(baseline: pd.DataFrame) -> pd.DataFrame:
    df = baseline.copy()

    def _tercile(s: pd.Series) -> pd.Series:
        if s.nunique() < 3 or len(s) < 6:
            return pd.Series(["mid"] * len(s), index=s.index)
        return pd.qcut(s.rank(method="first"), 3, labels=ABSOLUTE_TIER_LABELS)

    df["tier_relative"] = (
        df.groupby("primary_language")["baseline_churn"].transform(_tercile).astype(str)
    )
    df["tier_absolute"] = pd.cut(
        df["baseline_churn"], bins=ABSOLUTE_TIER_BINS, labels=ABSOLUTE_TIER_LABELS
    ).astype(str)
    return df


def _group_mean_diff(df: pd.DataFrame, pre_col: str, post_col: str, unit: str) -> dict:
    """Generic plain-units before/after by AI-likelihood group: a difference
    of group means/medians, safe from the near-zero-denominator blowup a %
    change has when the grouping variable correlates with the baseline level
    (a difference, not a ratio, so it stays sane for any feature). Reports
    both mean and median since several tables in this study show a real
    mean/median divergence (a few extreme individuals can drag a mean far from
    what's typical) -- presenting only one would be misleading. Shared by the
    churn before/after and every stylometry/commit-size feature below."""
    if df.empty or "p_ai" not in df.columns:
        return {}
    d = df.dropna(subset=["p_ai", pre_col, post_col])
    groups = {
        "likely_ai": d[d["p_ai"] >= AI_GROUP_THRESHOLD],
        "unlikely_ai": d[d["p_ai"] < AI_GROUP_THRESHOLD],
    }
    out: dict[str, dict] = {}
    for gname, gdf in groups.items():
        if gdf.empty:
            out[gname] = {"n": 0}
            continue
        pre = gdf[pre_col].to_numpy()
        post = gdf[post_col].to_numpy()
        rng = np.random.default_rng(17)
        n = len(pre)
        idx = np.arange(n)
        diffs = np.empty(2000)
        for b in range(2000):
            s = rng.choice(idx, size=n, replace=True)
            diffs[b] = post[s].mean() - pre[s].mean()
        out[gname] = {
            "n": int(n),
            f"mean_pre_{unit}": round(float(pre.mean()), 3),
            f"mean_post_{unit}": round(float(post.mean()), 3),
            f"mean_absolute_diff_{unit}": round(float(post.mean() - pre.mean()), 3),
            "mean_diff_ci_lo": round(float(np.percentile(diffs, 2.5)), 3),
            "mean_diff_ci_hi": round(float(np.percentile(diffs, 97.5)), 3),
            f"median_pre_{unit}": round(float(np.median(pre)), 3),
            f"median_post_{unit}": round(float(np.median(post)), 3),
            f"median_absolute_diff_{unit}": round(float(np.median(post) - np.median(pre)), 3),
        }
    return out


def _absolute_diff_by_group(before_after: pd.DataFrame) -> dict:
    return _group_mean_diff(
        before_after, "pre_churn_per_month", "post_churn_per_month", "lines_per_month"
    )


def _tier_breakdown(
    dev_dose: pd.DataFrame, before_after: pd.DataFrame, baseline: pd.DataFrame, tier_col: str
) -> dict:
    """Dose-response slope and a log-excess group comparison, computed *within*
    each productivity tier rather than pooled -- the direct test of "does AI
    help the second row more than the already-hyperactive".

    Deliberately does NOT reuse the plain-units before/after (percentage-of-
    baseline) comparison here: the tier itself is defined by pre-anchor
    baseline churn, so the "low" tier has a baseline close to zero by
    construction, and dividing by it produces exploding, meaningless percentages
    (observed in practice: five-figure "% changes" for the low tier -- a
    denominator artifact, not a finding). ``dev_dose``'s ``post_excess`` is a
    residual against each person's *own* fitted trend (bounded via
    RESID_CLIP), not a ratio against their level, so it stays well-behaved
    regardless of which tier -- by baseline level -- is being looked at.
    """
    dose = dev_dose.merge(baseline[["repo_id", "dev", tier_col]], on=["repo_id", "dev"], how="left")
    ba = before_after.merge(baseline[["repo_id", "dev", tier_col]], on=["repo_id", "dev"], how="left")

    out: dict[str, dict] = {}
    for tier in ABSOLUTE_TIER_LABELS:
        tdose = dose[(dose[tier_col] == tier) & dose["p_ai"].notna()]
        tba = ba[ba[tier_col] == tier]
        entry: dict[str, object] = {"n_dose": int(len(tdose))}
        if len(tdose) >= 5:
            X = np.column_stack([np.ones(len(tdose)), tdose["p_ai"].to_numpy()])
            y = tdose["post_excess"].to_numpy()
            res = ols_cluster(X, y, ["const", "p_ai"], groups=tdose["repo_id"].to_numpy())
            s = res.summary()[1]
            entry["dose_response_slope"] = s["coef"]
            entry["dose_response_slope_p"] = s["p"]
        entry["log_excess_by_p_ai_group"] = _dev_heterogeneity(tdose) if not tdose.empty else {}
        entry["absolute_lines_per_month_by_p_ai_group"] = (
            _absolute_diff_by_group(tba) if not tba.empty else {}
        )
        out[tier] = entry
    return out


def _dev_placebo_tiers(am: pd.DataFrame, meta: pd.DataFrame, prop_auth: pd.DataFrame) -> dict:
    """Placebo for the tier-stratified dose-response: fake anchor, pre-real-
    anchor data only, tiers reassigned using the *fake* anchor's own
    pre-period baseline (the fair parallel-construction analogue of the real
    tier assignment). If a tier shows a "significant" slope here too, the real
    result for that tier is a method artifact, not evidence of an AI effect --
    this is the check that must be run before trusting the low-tier finding.
    """
    if prop_auth.empty:
        return {}
    plac = _ord(PLACEBO_ANCHOR)
    df = am.copy()
    df["t"] = df["ym"].map(_ord) - plac
    df = df[df["ym"] < InclusionRule().anchor]  # pre-real-anchor only, as in _dev_placebo

    dose = _dev_counterfactual(df)
    if dose.empty:
        return {}
    post = dose[(dose["t"] >= 0) & (dose["t"] <= DEV_POST_WINDOW)]
    per_dev = post.groupby(["repo_id", "dev"], as_index=False)["resid"].mean().rename(
        columns={"resid": "post_excess"}
    )
    per_dev = per_dev.merge(
        prop_auth[["repo_id", "dev", "p_ai"]], on=["repo_id", "dev"], how="left"
    ).dropna(subset=["p_ai"])

    placebo_baseline = _assign_tiers(_dev_baseline(df, meta))

    out: dict[str, dict] = {}
    for tier_col in ("tier_relative", "tier_absolute"):
        merged = per_dev.merge(
            placebo_baseline[["repo_id", "dev", tier_col]], on=["repo_id", "dev"], how="left"
        )
        scheme: dict[str, dict] = {}
        for tier in ABSOLUTE_TIER_LABELS:
            t = merged[merged[tier_col] == tier]
            entry: dict[str, object] = {"n": int(len(t))}
            if len(t) >= 5:
                X = np.column_stack([np.ones(len(t)), t["p_ai"].to_numpy()])
                y = t["post_excess"].to_numpy()
                res = ols_cluster(X, y, ["const", "p_ai"], groups=t["repo_id"].to_numpy())
                s = res.summary()[1]
                entry["slope"] = s["coef"]
                entry["slope_p"] = s["p"]
            scheme[tier] = entry
        out[tier_col] = scheme
    return out


# ---------------------------------------------------------------------------
# cross-repo developer aggregation: the pseudonymised dev hash is already
# stable across repos (same salted email -> same hash), so a contributor
# active in several of our sampled repos can be tracked as one person across
# all of them rather than as separate repo-dev observations. With a broad
# random sample of repos this overlap is usually thin (most contributors only
# appear in one sampled repo); it becomes much more powerful with a
# deliberately-connected sample (e.g. a single ecosystem's core maintainers,
# who by construction work across many of that ecosystem's repos).
# ---------------------------------------------------------------------------


def _cross_repo_overlap(am: pd.DataFrame) -> dict:
    per_dev = am.groupby("dev")["repo_id"].nunique()
    return {
        "unique_devs": int(per_dev.shape[0]),
        "devs_in_multiple_repos": int((per_dev > 1).sum()),
        "share_in_multiple_repos": round(float((per_dev > 1).mean()), 4) if len(per_dev) else 0.0,
        "max_repos_by_one_dev": int(per_dev.max()) if len(per_dev) else 0,
    }


def _dev_counterfactual_pooled(am: pd.DataFrame, min_repos: int = 2) -> pd.DataFrame:
    """Same idea as _dev_counterfactual, but keyed by dev alone: churn summed
    across every sampled repo that developer touches each month, for the
    subset active in >= min_repos of them. Only informative when cross-repo
    overlap is non-trivial (see _cross_repo_overlap); the caller should check
    that before relying on this."""
    per_dev_repos = am.groupby("dev")["repo_id"].nunique()
    multi = set(per_dev_repos[per_dev_repos >= min_repos].index)
    if not multi:
        return pd.DataFrame(columns=["dev", "t", "resid", "pct"])
    pooled = (
        am[am["dev"].isin(multi)]
        .groupby(["dev", "ym", "t"], as_index=False)["churn"].sum()
    )
    rows = []
    for dev, g in pooled.groupby("dev"):
        g = g.sort_values("t")
        pre, post = g[g["t"] < 0], g[g["t"] >= 0]
        if len(pre) < DEV_MIN_PRE or len(post) < DEV_MIN_POST:
            continue
        Xpre = _dev_design(pre)
        ypre = np.log1p(pre["churn"].to_numpy())
        beta = np.linalg.pinv(Xpre.T @ Xpre) @ (Xpre.T @ ypre)
        Xall = _dev_design(g)
        resid = np.clip(np.log1p(g["churn"].to_numpy()) - Xall @ beta, -RESID_CLIP, RESID_CLIP)
        rows.append(pd.DataFrame({"dev": dev, "t": g["t"].to_numpy(), "resid": resid,
                                  "pct": np.expm1(resid)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["dev", "t", "resid", "pct"]
    )


# ---------------------------------------------------------------------------
# portfolio breadth: does AI let someone spread across more concurrent
# projects, rather than (or in addition to) producing more within one? The
# per-repo churn analyses above cannot see this by construction -- each is
# scoped to a single repo at a time. This is a plain count, not a ratio of
# baseline-dependent quantities, so unlike the tier before/after work it does
# not have a near-zero-denominator failure mode; still compared via a
# bootstrap on the mean difference rather than a naive % change, since a
# handful of very-high-breadth individuals (bots, prolific maintainers) could
# otherwise dominate a plain mean.
# ---------------------------------------------------------------------------


def _dev_portfolio_breadth(am: pd.DataFrame, prop_auth: pd.DataFrame) -> pd.DataFrame:
    """Per unique developer, pooled across every sampled repo they touch: how
    many distinct repos were they concurrently active in per month, pre- vs
    post-anchor."""
    pre, post = am[am["t"] < 0], am[am["t"] >= 0]
    pre_breadth = pre.groupby(["dev", "ym"])["repo_id"].nunique().groupby("dev").mean()
    post_breadth = post.groupby(["dev", "ym"])["repo_id"].nunique().groupby("dev").mean()
    lifetime_pre = pre.groupby("dev")["repo_id"].nunique()
    lifetime_post = post.groupby("dev")["repo_id"].nunique()

    df = pd.DataFrame({
        "avg_monthly_repos_pre": pre_breadth, "avg_monthly_repos_post": post_breadth,
        "lifetime_repos_pre": lifetime_pre, "lifetime_repos_post": lifetime_post,
    }).dropna(subset=["avg_monthly_repos_pre", "avg_monthly_repos_post"]).reset_index()

    if not prop_auth.empty:
        # p_ai is estimated per (repo_id, dev); take the max across a person's
        # repos as a simple pooled proxy (signal from any one repo counts).
        dev_p_ai = prop_auth.groupby("dev", as_index=False)["p_ai"].max()
        df = df.merge(dev_p_ai, on="dev", how="left")
    return df


def _breadth_by_group(breadth: pd.DataFrame) -> dict:
    if breadth.empty or "p_ai" not in breadth.columns:
        return {}
    df = breadth.dropna(subset=["p_ai"])
    groups = {
        "likely_ai": df[df["p_ai"] >= AI_GROUP_THRESHOLD],
        "unlikely_ai": df[df["p_ai"] < AI_GROUP_THRESHOLD],
    }
    out: dict[str, dict] = {}
    for gname, gdf in groups.items():
        if gdf.empty:
            out[gname] = {"n": 0}
            continue
        pre = gdf["avg_monthly_repos_pre"].to_numpy()
        post = gdf["avg_monthly_repos_post"].to_numpy()
        rng = np.random.default_rng(13)
        n = len(pre)
        diffs = np.empty(2000)
        idx = np.arange(n)
        for b in range(2000):
            s = rng.choice(idx, size=n, replace=True)
            diffs[b] = post[s].mean() - pre[s].mean()
        out[gname] = {
            "n": int(n),
            "mean_avg_monthly_repos_pre": round(float(pre.mean()), 2),
            "mean_avg_monthly_repos_post": round(float(post.mean()), 2),
            "mean_diff": round(float(post.mean() - pre.mean()), 3),
            "mean_diff_lo": round(float(np.percentile(diffs, 2.5)), 3),
            "mean_diff_hi": round(float(np.percentile(diffs, 97.5)), 3),
            "share_breadth_increased": round(float((post > pre).mean()), 3),
        }
    return out


# ---------------------------------------------------------------------------
# commit-message stylometry & commit-size spread: behavioral fingerprints
# independent of any trailer or self-disclosure (concept.md sec. 3/2.4).
# Simple mean-based before/after per feature -- these are plain, roughly
# well-scaled quantities (character counts, ratios in [0,1], a size std-dev),
# not log-residuals, so the shared _group_mean_diff (a difference of means, not
# a ratio) is the right tool, with its own placebo using the same fake-anchor,
# pre-real-anchor-data-only discipline as the churn placebo.
# ---------------------------------------------------------------------------

STYLE_FEATURES = ("msg_chars", "msg_words", "msg_has_bullets", "msg_unique_word_ratio", "size_std")


def _dev_feature_before_after(
    am: pd.DataFrame, feature_col: str, prop_auth: pd.DataFrame
) -> pd.DataFrame:
    """Per developer: mean of an arbitrary monthly feature column, pre- vs
    post-anchor. Same continuity floor as the churn before/after table."""
    rows = []
    for (rid, dev), g in am.groupby(["repo_id", "dev"]):
        pre, post = g[g["t"] < 0], g[g["t"] >= 0]
        if len(pre) < DEV_MIN_PRE or len(post) < DEV_MIN_POST:
            continue
        rows.append({
            "repo_id": rid, "dev": dev,
            f"pre_{feature_col}": float(pre[feature_col].mean()),
            f"post_{feature_col}": float(post[feature_col].mean()),
        })
    df = pd.DataFrame(rows)
    if df.empty or prop_auth.empty:
        return df
    return df.merge(
        prop_auth[["repo_id", "dev", "p_ai", "label_source"]], on=["repo_id", "dev"], how="left"
    )


def _dev_feature_placebo(am: pd.DataFrame, feature_col: str, prop_auth: pd.DataFrame) -> dict:
    """Placebo for one stylometry/size feature: fake anchor, pre-real-anchor
    data only. The group-diff here should be near zero; if it isn't, a real
    finding for this feature at the true anchor cannot be trusted."""
    if prop_auth.empty:
        return {}
    plac = _ord(PLACEBO_ANCHOR)
    df = am.copy()
    df["t"] = df["ym"].map(_ord) - plac
    df = df[df["ym"] < InclusionRule().anchor]
    ba = _dev_feature_before_after(df, feature_col, prop_auth)
    if ba.empty:
        return {}
    return _group_mean_diff(ba, f"pre_{feature_col}", f"post_{feature_col}", feature_col)


def _feature_battery(am: pd.DataFrame, features: tuple[str, ...], prop_auth: pd.DataFrame) -> dict:
    """Run the same before/after-by-group-plus-placebo recipe over a list of
    plain, well-scaled monthly feature columns. Shared by stylometry, language
    breadth, and time-of-day/weekday patterns below -- they differ only in
    which columns they point this at."""
    out: dict[str, dict] = {}
    for feat in features:
        if feat not in am.columns:
            continue
        ba = _dev_feature_before_after(am, feat, prop_auth)
        by_group = _group_mean_diff(ba, f"pre_{feat}", f"post_{feat}", feat) if not ba.empty else {}
        placebo = _dev_feature_placebo(am, feat, prop_auth)
        out[feat] = {"n": int(len(ba)), "by_p_ai_group": by_group, "placebo_by_p_ai_group": placebo}
    return out


TIME_PATTERN_FEATURES = ("weekend_share", "evening_share")
LANGUAGE_BREADTH_FEATURES = ("n_langs",)


def _bugfix_feature_share(am: pd.DataFrame) -> pd.DataFrame:
    """Derived per (repo, dev, ym): share of classified commits that read as a
    bugfix vs. new-feature work (see stylometry.classify_commit). NaN where
    nothing was classified that month (left out of the mean, not treated as 0
    -- a silent month should not count as "no bugfixes")."""
    df = am.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["feature_share"] = np.where(
            df["total_commits"] > 0, df["feature_commits"] / df["total_commits"], np.nan
        )
        df["bugfix_share"] = np.where(
            df["total_commits"] > 0, df["bugfix_commits"] / df["total_commits"], np.nan
        )
    return df


BUGFIX_FEATURE_COLUMNS = ("feature_share", "bugfix_share")


# ---------------------------------------------------------------------------
# first-time contributor rate: does AI availability coincide with more new
# people joining a project, not just existing contributors doing more? A
# repo-level metric (onboarding is a property of the project, not of one
# individual) -- descriptive; not placebo-tested here since, unlike the
# per-entity counterfactuals above, there is no natural single-entity trend to
# extrapolate for a "share of newcomers" quantity.
# ---------------------------------------------------------------------------


def _first_contributor_rate(am_inc: pd.DataFrame) -> dict:
    if am_inc.empty:
        return {}
    df = am_inc.sort_values(["repo_id", "dev", "t"]).copy()
    first_t = df.groupby(["repo_id", "dev"])["t"].transform("min")
    df["is_first_month"] = df["t"] == first_t
    per_month = (
        df.groupby(["repo_id", "ym", "t"])
        .agg(active=("dev", "nunique"), first_time=("is_first_month", "sum"))
        .reset_index()
    )
    per_month["first_time_share"] = per_month["first_time"] / per_month["active"]
    pre = per_month.loc[per_month["t"] < 0, "first_time_share"].to_numpy()
    post = per_month.loc[per_month["t"] >= 0, "first_time_share"].to_numpy()
    if len(pre) == 0 or len(post) == 0:
        return {}
    rng = np.random.default_rng(23)
    diffs = np.empty(2000)
    for b in range(2000):
        ps = rng.choice(pre, size=len(pre), replace=True)
        qs = rng.choice(post, size=len(post), replace=True)
        diffs[b] = qs.mean() - ps.mean()
    return {
        "n_repo_months_pre": int(len(pre)),
        "n_repo_months_post": int(len(post)),
        "mean_first_time_share_pre": round(float(pre.mean()), 4),
        "mean_first_time_share_post": round(float(post.mean()), 4),
        "mean_diff": round(float(post.mean() - pre.mean()), 4),
        "mean_diff_ci_lo": round(float(np.percentile(diffs, 2.5)), 4),
        "mean_diff_ci_hi": round(float(np.percentile(diffs, 97.5)), 4),
    }


# ---------------------------------------------------------------------------
# "comeback" / reactivation: developers who went quiet for a long stretch and
# then resumed. Deliberately uses the FULL (not repo-continuity-gated)
# developer population: an individual can go quiet and return even within a
# repo whose team stays continuously active overall, and repos excluded from
# the main panel *because* of a long gap are exactly the population this
# question is about. Purely descriptive -- not placebo-tested (there is no
# single-entity trend to extrapolate for "did a gap end"), and explicitly
# normalised per calendar month of each era since the pre- and post-anchor
# windows in this dataset are different lengths.
# ---------------------------------------------------------------------------

REACTIVATION_GAP_MONTHS = 6


def _reactivation_stats(am: pd.DataFrame, prop_auth: pd.DataFrame) -> dict:
    if am.empty:
        return {}
    rows = []
    for (rid, dev), g in am.groupby(["repo_id", "dev"]):
        months_t = sorted(g["t"].unique())
        for i in range(1, len(months_t)):
            gap = months_t[i] - months_t[i - 1] - 1
            if gap >= REACTIVATION_GAP_MONTHS:
                rows.append({
                    "repo_id": rid, "dev": dev, "gap_months": int(gap),
                    "resumption_t": int(months_t[i]),
                })
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        return {"n_gap_episodes": 0}

    pre = episodes[episodes["resumption_t"] < 0]
    post = episodes[episodes["resumption_t"] >= 0]
    t_min, t_max = int(am["t"].min()), int(am["t"].max())
    pre_era_months = max(1, 0 - t_min)
    post_era_months = max(1, t_max + 1)

    out: dict[str, object] = {
        "n_gap_episodes": int(len(episodes)),
        "gap_threshold_months": REACTIVATION_GAP_MONTHS,
        "resumptions_pre_anchor": int(len(pre)),
        "resumptions_post_anchor": int(len(post)),
        "pre_era_months": pre_era_months,
        "post_era_months": post_era_months,
        "resumptions_per_month_pre": round(len(pre) / pre_era_months, 4),
        "resumptions_per_month_post": round(len(post) / post_era_months, 4),
    }
    if not prop_auth.empty:
        ep = episodes.merge(
            prop_auth[["repo_id", "dev", "p_ai"]], on=["repo_id", "dev"], how="left"
        )
        ep_post = ep[(ep["resumption_t"] >= 0) & ep["p_ai"].notna()]
        if not ep_post.empty:
            out["post_anchor_resumers_by_p_ai_group"] = {
                "likely_ai": int((ep_post["p_ai"] >= AI_GROUP_THRESHOLD).sum()),
                "unlikely_ai": int((ep_post["p_ai"] < AI_GROUP_THRESHOLD).sum()),
            }
    return out


# ---------------------------------------------------------------------------
# DEDICATED, STANDALONE analyses (deliberately kept separate from the p_ai /
# dose-response / tier pipeline above, per explicit project direction):
#
# 1. Raw AI-tool mention rate over calendar time -- a "hype curve" built from
#    bare-word mentions (see stylometry.mentions), not the strict Tier-1
#    attribution patterns. Purely descriptive; never merged into any group
#    comparison.
#
# 2. Per-developer commit-message-length changepoint detection. The key
#    reasoning for keeping this OUT of the main pipeline: someone who
#    deliberately omits a Co-Authored-By trailer specifically so their AI use
#    isn't flagged is, by construction, sanitising exactly the signal Design D
#    relies on -- but their *writing style* can still shift, unintentionally,
#    around the same time. Folding a style-based signal into the same
#    composite score that already privileges trailer-based labels would let
#    the two channels contaminate each other and obscure exactly the
#    "self-sanitising user" case that makes this signal valuable in the first
#    place. Reported here as its own section: a longer message is suggestive,
#    never proof -- an unrelated habit change (new job, adopting a commit
#    style guide, general growth as a writer) produces an identical pattern.
# ---------------------------------------------------------------------------


def mentions_timeseries(mentions_month: pd.DataFrame) -> pd.DataFrame:
    if mentions_month.empty:
        return mentions_month
    term_cols = [c for c in mentions_month.columns if c.startswith("mentions_")]
    if not term_cols:
        return pd.DataFrame()
    # Defensive display-level clip: a bogus future system clock at commit time
    # (seen for real: zebra-rs/zebra-rs, dated 2106/2242) survives in data
    # gathered before miner.py's max_ym guard existed. A single such commit
    # would otherwise corrupt "latest month" reporting here.
    now_ym = datetime.now(timezone.utc).strftime("%Y-%m")
    mentions_month = mentions_month[mentions_month["ym"] <= now_ym]
    agg = {c: "sum" for c in term_cols}
    agg["total_commits"] = "sum"
    g = mentions_month.groupby("ym", as_index=False).agg(agg)
    g["any_mention"] = g[term_cols].sum(axis=1)
    g["mention_rate"] = g["any_mention"] / g["total_commits"].replace(0, np.nan)
    return g.sort_values("ym").reset_index(drop=True)


CHANGEPOINT_MIN_MONTHS = 8  # need enough active months either side of a candidate split
CHANGEPOINT_MIN_MARGIN = 3
CHANGEPOINT_EFFECT_THRESHOLD = 1.0  # pooled-std-normalised jump size counted as "strong"
# Every developer's series ends around the same real gather date. A split
# right at that shared edge is cheap to "find" purely from the small-sample
# variance of a short trailing segment, not real signal -- confirmed on this
# project's own data: with margin=3, 57% of breakpoints detected in the most
# recent quarter sat within 3 months of that developer's own last month,
# vs. 7% for older breakpoints, and the population histogram showed an
# implausible ~6x spike in the two months right before the gather date.
# Trim the shared most-recent months globally before searching so no
# breakpoint can be an artifact of that common right-censoring boundary.
CHANGEPOINT_TRIM_RECENT_MONTHS = 6


CHANGEPOINT_N_PERM = 99
CHANGEPOINT_P_THRESHOLD = 0.05


def _best_split_gap(vals: np.ndarray, margin: int) -> tuple[float, int]:
    """Vectorised: for every split point i in [margin, n-margin), the
    after-mean minus before-mean gap, via cumulative sums (no Python loop
    over split points). Returns the best (gap, index)."""
    n = len(vals)
    cs = np.cumsum(vals)
    total = cs[-1]
    idx = np.arange(margin, n - margin)
    before_mean = cs[idx - 1] / idx
    after_mean = (total - cs[idx - 1]) / (n - idx)
    gaps = after_mean - before_mean
    pos = int(np.argmax(gaps))
    return float(gaps[pos]), int(idx[pos])


def detect_style_changepoints(am: pd.DataFrame, feature_col: str = "msg_chars") -> pd.DataFrame:
    """Per (repo, dev): the single split point that maximises the before/after
    gap in a stylometric feature -- a transparent two-segment changepoint
    (not a full Bayesian changepoint model, proportionate to how sparse and
    noisy an individual's monthly series is here). Reports the breakpoint's
    own calendar month, not anchor-relative t, so the population distribution
    of detected breakpoints can be checked against known AI-tool release
    dates independently of this project's own calendar-anchor choice.

    IMPORTANT: searching every possible split point and keeping the best gap
    is a max-over-many-candidates statistic -- even pure noise produces a
    biggish-looking "best" gap this way (confirmed on this project's own
    first pass: ~31% of developers came back as "strong" jumps at a fixed
    effect-size cutoff, an implausibly high rate). Each developer's own
    observed gap is therefore checked against a permutation null (their own
    monthly values, shuffled, re-run through the same best-split search) --
    `perm_p` is the resulting p-value; only rows with `perm_p` below
    CHANGEPOINT_P_THRESHOLD should be read as a real, not noise-explainable,
    structural break.
    """
    if feature_col not in am.columns or am.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(29)
    cutoff_t = int(am["t"].max()) - CHANGEPOINT_TRIM_RECENT_MONTHS
    rows = []
    for (rid, dev), g in am[am["t"] <= cutoff_t].groupby(["repo_id", "dev"]):
        g = g.sort_values("t")
        n = len(g)
        if n < CHANGEPOINT_MIN_MONTHS:
            continue
        vals = g[feature_col].to_numpy(dtype=float)
        yms = g["ym"].to_numpy()
        ts = g["t"].to_numpy()
        m = CHANGEPOINT_MIN_MARGIN
        best_gap, best_idx = _best_split_gap(vals, m)

        exceed = 1  # +1 correction: the observed sample counts as one draw
        for _ in range(CHANGEPOINT_N_PERM):
            shuffled = rng.permutation(vals)
            perm_gap, _ = _best_split_gap(shuffled, m)
            if perm_gap >= best_gap:
                exceed += 1
        perm_p = exceed / (CHANGEPOINT_N_PERM + 1)

        before, after = vals[:best_idx], vals[best_idx:]
        pooled_var = (
            (before.var(ddof=1) + after.var(ddof=1)) / 2
            if len(before) > 1 and len(after) > 1 else 0.0
        )
        effect = float(best_gap / np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
        rows.append({
            "repo_id": rid, "dev": dev,
            "breakpoint_ym": str(yms[best_idx]), "breakpoint_t": int(ts[best_idx]),
            "gap": round(float(best_gap), 2),
            "effect_size": round(effect, 3),
            "perm_p": round(float(perm_p), 4),
            "before_mean": round(float(before.mean()), 2),
            "after_mean": round(float(after.mean()), 2),
            "n_months": int(n),
        })
    return pd.DataFrame(rows)


def style_changepoint_histogram(changepoints: pd.DataFrame) -> pd.DataFrame:
    """Population histogram of detected breakpoint months, restricted to jumps
    that clear BOTH a minimum effect size and the permutation-null p-value
    (see detect_style_changepoints' docstring for why the latter matters) --
    the shape to visually compare against known model-release dates
    (concept.md sec. 7)."""
    if changepoints.empty or "perm_p" not in changepoints.columns:
        return pd.DataFrame(columns=["breakpoint_ym", "n_breakpoints"])
    strong = changepoints[
        (changepoints["effect_size"] >= CHANGEPOINT_EFFECT_THRESHOLD)
        & (changepoints["perm_p"] < CHANGEPOINT_P_THRESHOLD)
    ]
    if strong.empty:
        return pd.DataFrame(columns=["breakpoint_ym", "n_breakpoints"])
    return (
        strong.groupby("breakpoint_ym").size().reset_index(name="n_breakpoints")
        .sort_values("breakpoint_ym").reset_index(drop=True)
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
    prop_auth = pd.DataFrame()
    if not afeats.empty:
        prop_auth = _pu_propensity(afeats, "dev_has_sig", ["repo_id", "dev"], n_boot=100)
        save_table(prop_auth, out, "propensity-author")

    before_after = _before_after_repo(rmt_inc, meta, prop_repo)
    ba_groups: dict = {}
    if not before_after.empty:
        save_table(before_after, out, "before-after")
        ba_groups = _before_after_groups(before_after)

    # -- developer-level analysis (only within included, continuity-gated repos:
    #    the same data-quality concerns that exclude a repo make its individual
    #    developer histories unreliable too) --------------------------------
    am_inc = am[am["repo_id"].isin(inc_ids)]
    dev_summary: dict = {}
    dev_excess = _dev_counterfactual(am_inc)
    if not dev_excess.empty:
        dev_es = _event_study(dev_excess)
        save_table(dev_es, out, "dev-event-study")

        dev_dose, dev_dose_stat = _dev_dose_response(dev_excess, prop_auth)
        if not dev_dose.empty:
            save_table(dev_dose, out, "dev-dose-response")

        dev_placebo = _dev_placebo(am_inc)

        dev_before_after = _before_after_dev(am_inc, prop_auth)
        dev_ba_groups: dict = {}
        dev_absolute_diff: dict = {}
        if not dev_before_after.empty:
            save_table(dev_before_after, out, "dev-before-after")
            dev_ba_groups = _before_after_groups(
                dev_before_after, "pre_churn_per_month", "post_churn_per_month"
            )
            dev_absolute_diff = _absolute_diff_by_group(dev_before_after)

        heterogeneity = _dev_heterogeneity(dev_dose) if not dev_dose.empty else {}
        top_gainers = _dev_top_gainers(dev_dose) if not dev_dose.empty else pd.DataFrame()
        if not top_gainers.empty:
            save_table(top_gainers, out, "dev-top-gainers", csv=True)

        overlap = _cross_repo_overlap(am_inc)
        pooled_summary: dict = {}
        pooled_excess = _dev_counterfactual_pooled(am_inc)
        if not pooled_excess.empty:
            pooled_es = _event_study(pooled_excess.rename(columns={"dev": "repo_id"}))
            save_table(pooled_es, out, "dev-pooled-event-study")
            pooled_post = pooled_es[(pooled_es["t"] >= 0) & (pooled_es["t"] <= DEV_POST_WINDOW)]
            pooled_summary = {
                "n_devs": int(pooled_excess["dev"].nunique()),
                "mean_post_excess": (
                    float(pooled_post["mean_excess"].mean()) if len(pooled_post) else None
                ),
            }

        baseline = _dev_baseline(am_inc, meta)
        baseline = _assign_tiers(baseline)
        save_table(baseline, out, "dev-productivity-tiers")
        tiers_relative = (
            _tier_breakdown(dev_dose, dev_before_after, baseline, "tier_relative")
            if not dev_dose.empty else {}
        )
        tiers_absolute = (
            _tier_breakdown(dev_dose, dev_before_after, baseline, "tier_absolute")
            if not dev_dose.empty else {}
        )
        tiers_placebo = _dev_placebo_tiers(am_inc, meta, prop_auth)

        breadth = _dev_portfolio_breadth(am_inc, prop_auth)
        if not breadth.empty:
            save_table(breadth, out, "dev-portfolio-breadth")
        breadth_groups = _breadth_by_group(breadth)

        stylometry = _feature_battery(am_inc, STYLE_FEATURES, prop_auth)
        language_breadth = _feature_battery(am_inc, LANGUAGE_BREADTH_FEATURES, prop_auth)
        time_patterns = _feature_battery(am_inc, TIME_PATTERN_FEATURES, prop_auth)
        bugfix_am = _bugfix_feature_share(am_inc)
        bugfix_feature = _feature_battery(bugfix_am, BUGFIX_FEATURE_COLUMNS, prop_auth)

        first_contrib = _first_contributor_rate(am_inc)
        # Reactivation deliberately uses the FULL developer population (not
        # am_inc): repos excluded from the main panel for a long gap are
        # exactly the population the comeback question is about.
        reactivation = _reactivation_stats(am, prop_auth)

        dev_post_es = dev_es[(dev_es["t"] >= 0) & (dev_es["t"] <= DEV_POST_WINDOW)]["mean_excess"]
        dev_summary = {
            "n_devs_analyzed": int(dev_excess[["repo_id", "dev"]].drop_duplicates().shape[0]),
            "mean_post_excess": float(dev_post_es.mean()) if len(dev_post_es) else None,
            "dose_response_by_own_p_ai": dev_dose_stat,
            "placebo_mean_excess": dev_placebo["mean_excess"],
            "before_after_by_p_ai_group": dev_ba_groups,
            "absolute_lines_per_month_by_p_ai_group": dev_absolute_diff,
            "heterogeneity_by_p_ai_group": heterogeneity,
            "cross_repo_overlap": overlap,
            "cross_repo_pooled": pooled_summary,
            "productivity_tiers": {
                "relative_within_language": tiers_relative,
                "absolute_bands": tiers_absolute,
                "placebo": tiers_placebo,
            },
            "portfolio_breadth_by_p_ai_group": breadth_groups,
            "stylometry": stylometry,
            "language_breadth": language_breadth,
            "time_patterns": time_patterns,
            "bugfix_feature_share": bugfix_feature,
            "first_contributor_rate": first_contrib,
            "reactivation": reactivation,
        }
        (out / "dev-placebo.json").write_text(json.dumps(dev_placebo, indent=2))

    # -- dedicated, standalone analyses (see the section docstring above for
    #    why these are deliberately NOT part of developer_level) -----------
    dedicated: dict = {}
    try:
        mentions_month = load_table(panels, "mentions-month")
    except FileNotFoundError:
        mentions_month = pd.DataFrame()
    mts = mentions_timeseries(mentions_month)
    if not mts.empty:
        save_table(mts, out, "mentions-timeseries")
        dedicated["mentions_timeseries"] = {
            "n_months": int(len(mts)),
            "peak_month": str(mts.loc[mts["mention_rate"].idxmax(), "ym"]) if mts["mention_rate"].notna().any() else None,
            "peak_rate": float(mts["mention_rate"].max()) if mts["mention_rate"].notna().any() else None,
            "latest_month": str(mts["ym"].iloc[-1]),
            "latest_rate": float(mts["mention_rate"].iloc[-1]) if pd.notna(mts["mention_rate"].iloc[-1]) else None,
        }

    changepoints = detect_style_changepoints(am_inc, "msg_chars") if "msg_chars" in am_inc.columns else pd.DataFrame()
    if not changepoints.empty:
        save_table(changepoints, out, "style-changepoints")
        hist = style_changepoint_histogram(changepoints)
        if not hist.empty:
            save_table(hist, out, "style-changepoint-histogram")
        strong_mask = (
            (changepoints["effect_size"] >= CHANGEPOINT_EFFECT_THRESHOLD)
            & (changepoints["perm_p"] < CHANGEPOINT_P_THRESHOLD)
        )
        dedicated["style_changepoints"] = {
            "n_devs_evaluated": int(len(changepoints)),
            "n_strong_jumps": int(strong_mask.sum()),
            "n_strong_jumps_effect_only_uncorrected": int(
                (changepoints["effect_size"] >= CHANGEPOINT_EFFECT_THRESHOLD).sum()
            ),
            "effect_threshold": CHANGEPOINT_EFFECT_THRESHOLD,
            "perm_p_threshold": CHANGEPOINT_P_THRESHOLD,
            "note": "descriptive only -- a longer message is suggestive, never proof; "
                    "n_strong_jumps requires both an effect-size cutoff AND permutation "
                    "significance (max-over-split-points search otherwise overstates how "
                    "many 'jumps' are real, see detect_style_changepoints docstring); "
                    "see module docstring for why this is kept separate from p_ai/dose-response",
        }

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
        "before_after_by_ai_group": ba_groups,
        "developer_level": dev_summary,
        "dedicated_standalone_analyses": dedicated,
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
