# AI-Induced SLOC Velocity — Research Concept

Status: draft methodology. No code yet. Goal: a design that survives peer review
in empirical software engineering, not a plausible-looking dashboard.

## 0. The one-sentence thesis

*Does repo-level adoption of generative-AI coding tools cause a measurable,
non-spurious shift in code-production velocity and code characteristics?*

Note the wording: **repo-level adoption** (not calendar time) and **shift in
characteristics** (not just "more lines"). Both choices are what make the study
defensible. The rest of this document explains why.

## 1. Why the naive design fails (and must be rejected)

The obvious approach — average SLOC/week before ChatGPT vs. after — is not
publishable, for four independent reasons:

1. **No control group.** A global calendar cutoff (2022-11-30) treats every repo
   at once. Anything else that trended in 2022–2025 (repo growth with age, team
   growth, CI/tooling changes, macro dev-employment shifts) is perfectly
   confounded with "AI". You cannot separate them with a before/after mean.
2. **Unobserved treatment timing.** Calendar date is not adoption date. Teams
   adopt AI on a staggered, partly hidden schedule. Using a single global date
   introduces massive misclassification.
3. **SLOC is a discredited productivity metric.** More lines can mean bloat,
   generated code, vendored dependencies, or verbose scaffolding — the *opposite*
   of productivity. Any velocity finding must be paired with density/quality
   metrics or it is meaningless.
4. **Selection & survivorship.** Repos that are still active in 2025 are not a
   random sample of 2020 repos. Naive panels bake in survivorship bias.

Everything below exists to neutralize these four failures.

## 2. Identification strategy

**Design decision (revised 2026-07-03):** repo-level AI-attribution signatures
(`Co-Authored-By: Claude`, Copilot trailers, …) are **not usable as the treatment
variable**. They are systematically removed by squash-merges, rebases, cleanup
hooks, and plain opt-out. They are *positive-only and censored*: "present" ⇒ AI,
but "absent" ⇒ unknown. A treatment variable you can only observe when someone
chose not to delete it is worthless for DiD. So the identification does **not**
depend on observing per-repo adoption at all. Instead:

Primary designs need only a **calendar shock** (AI became available to everyone
at roughly known dates) plus a well-modeled **counterfactual**. Two co-primary,
plus two robustness designs. Agreement across them is the evidence.

### 2.1 Design A — Counterfactual "excess churn" / synthetic control (co-primary)

This is the "excess mortality" method the brief invoked, and it needs no adoption
marker. For each repo, fit its expected trajectory from the **pre-AI era**
(≤ 2021), conditioning on repo age, month-of-year seasonality, and team size.
Project it forward into the post-AI era and compute the **residual = actual −
expected** ("excess SLOC/churn"). Pool residuals across the whole panel.

- If AI has an effect, the pooled residual turns positive after the availability
  shock and *persists*; a one-off blip is not enough.
- Counterfactual model: per-repo Poisson/negative-binomial or log-linear trend +
  seasonal terms, or a Bayesian structural time series (BSTS / CausalImpact-style)
  using the not-yet-affected period as training. We fit *before* looking at the
  post-period residuals (pre-registered).
- **Placebo cutoffs** in the pre-period (e.g. pretend AI arrived in 2019) must
  yield ~zero excess. This replaces the missing control group.

### 2.2 Design B — Reduced-form event study on the availability shock (co-primary)

Treat AI *availability* (Section 7 dates) as an **intention-to-treat**: we don't
need to know who actually used it, only that the option appeared exogenously at a
known time for everyone. Estimate level/slope breaks in the aggregate and per-
stratum series around those dates, with placebo dates for the null distribution.
Reduced-form / ITT is a legitimate econometric answer to unobservable adoption:
the estimated effect is "effect of AI *becoming available*", which is exactly what
a policy-relevant claim should be, and it sidesteps the deleted-trailer problem
entirely.

### 2.3 The confounder defense that replaces the lost per-repo marker

Without a treatment marker we lose the clean control group, so we lean hard on
two design-based checks that are very hard for a generic secular trend to fake:

- **Dose-response by AI-suitability (key).** AI coding tools are far stronger for
  Python/JS/TS than for niche languages (COBOL, Fortran, Ada, Verilog, Nix, …).
  A genuine AI effect should be **larger where the tools are more capable** and
  near-zero where they are not. If the "AI jump" is uniform across all languages
  including those the models barely support, it is almost certainly a generic
  post-2022 trend, not AI. This gradient is our strongest identifying argument.
- **Negative-control outcomes.** Measure quantities AI should *not* move much but
  a generic "more activity" trend would (e.g. non-code churn: docs-only,
  config-only commits; issue/PR comment cadence if fetched). If these rise in
  lockstep with source churn, the source-churn rise is not AI-specific.

### 2.4 Design C — Within-developer / matched (elevated from robustness to co-primary)

Same developer, many repos, across the shock boundary. Developer fixed effects (or
a matched pre/post within-author panel) strip out "this person is just fast" and
compositional change (new hires). Isolates change *in the same hands*.

**Empirically validated (2026-07-04):** this stopped being merely a robustness
check once real data on ~300 repos showed *why* it matters concretely. AI
adoption is a per-developer choice, not a repo-level one, so repo-mean pooling
dilutes exactly the signal of interest; worse, it is confounded by **team-size
composition drift** — repos gain casual/drive-by contributors as they mature,
which mechanically lowers "churn per developer" with nothing to do with AI. In
the 309-repo / 156-included run, the repo-level placebo check came out at
+100-140% (badly failing), while the identical placebo re-run on the
developer-level counterfactual (same person's own churn, own trend, no
division by headcount) came out within ~2 percentage points of zero. Tracking
the same individual over time is immune to what happens to team composition
elsewhere; the repo-level pooling is not. Design C is therefore promoted to a
co-primary analysis (`aisloc.analysis.stats`'s developer-level section), run
alongside Designs A/B rather than only as a confirmatory check on them.

Implementation notes carried over from this validation:
- The dose is the developer's own PU-model `p_ai` (Design E), not language
  suitability — directly testing "do developers who probably use AI show a
  bigger change in their own output".
- Lifecycle curvature at the *per-entity* level (repo or developer) should be
  handled by keeping the trend **linear**, not quadratic: age/tenure and
  calendar time are the same straight line up to a shift for a single entity,
  so a quadratic term adds no real information, and naive polynomial
  extrapolation over a 24-30 month post window turned out to be numerically
  unstable in practice (it made both the repo- and developer-level placebo
  checks *worse*). Genuine age effects belong in the *pooled* cross-repo
  backbone regression instead, where age varies independently across entities.
- Because the effect is very unlikely to be uniform across people, population
  means are not the only thing worth reporting: dispersion/heterogeneity by
  AI-likelihood group (does the AI-likely group have a fatter right tail of
  large individual gains, even if the mean doesn't move?) and a standout-
  individuals leaderboard are computed alongside the mean dose-response.
- The pseudonymised developer hash is stable across repos, enabling cross-repo
  aggregation for contributors active in multiple sampled repos. This is thin
  under broad random sampling (~3% overlap in the 309-repo run) and much more
  powerful under a **deliberately connected sample** — one ecosystem's core
  maintainers, who work across many of that ecosystem's repos by construction
  (see `data/example-rust-core-repos.txt` and the README's "Developer-level
  analysis" section).

### 2.4b Design E — Per-entity AI-use propensity (PU learning)

Deliverable: a **calibrated probability that a given repo / developer uses AI**,
with a credible interval and the feature contributions. The honest framing is
**Positive-Unlabeled (PU) learning**: our only labels are *positives* (surviving
Tier-1 signatures, Section 3); "no signature" is *unlabeled*, not negative. So we
cannot train a naive classifier — we estimate `P(AI | features)` calibrated to an
assumed/estimated base rate (Elkan–Noto estimator, or a spy/nnPU approach), and
report results across a **range of base-rate assumptions** because the true
prevalence is unknown.

- Features (per repo and per author, all from the same mining pass): excess-churn
  residual vs. its own pre-AI baseline; shift in mean/percentile change-size;
  commit-cadence and burstiness change post-2022; comment-to-code shift; share of
  work in high-AI-suitability languages; boilerplate/rework ratio.
- Model: gradient-boosted trees or logistic PU with isotonic/Platt calibration;
  per-entity uncertainty via bootstrap over repos.
- Output columns: `p_ai`, `p_ai_lo`, `p_ai_hi` (CI), `label_source`
  (signature-positive / inferred), top SHAP-style feature attributions, and the
  base-rate assumption used. Written to `data/results/propensity-{repo,author}.csv`.
- **Honesty:** this is a propensity/likelihood, not proof. A signature-positive
  entity is `p_ai≈1` by construction; for the rest it is a model estimate whose
  reliability is bounded by the PU assumptions. We never present it as detection.
- Enabling requirement: gathering must aggregate churn **per author × month ×
  language**, not only per repo — so the schema carries an `authors` breakdown.

### 2.5 Design D — Staggered DiD on the validation subsample only (demoted)

Where Tier-1 signatures *do* survive, we still run Callaway–Sant'Anna staggered
DiD (Section 6) — but strictly as a **bounding/validation exercise on a biased
positive-only subsample**, never as the headline. Its role: does the direction
and rough magnitude of the excess-churn residual (Design A) agree with what we see
in the minority of repos where AI use is directly visible? Agreement corroborates;
it cannot, on its own, identify the population effect because the subsample is
self-selected (only teams that *didn't* strip trailers).

## 3. AI signatures — role after the revision

Signatures are demoted from "treatment variable" to a **positive-only validation
label**. We still mine them, because a noisy positive label is useful for
corroboration, but we never treat their absence as "no AI".

- Detected (positive only): `Co-Authored-By: Claude`, Copilot/aider/Cursor/
  Codeium trailers and bot authors, and AI-tool config artifacts entering the
  tree (`.cursor/`, `.aider*`, `.github/copilot-*`, Continue/Cody).
- Recorded per repo: `first_ai_signature_week`, `signal_class` — used **only** in
  Design D (validation subsample) and to describe adoption visibility, with an
  explicit caveat that it is a lower bound heavily depressed by trailer removal.
- We report the estimated *survival rate* of trailers where we can (e.g. repos
  that use both squash-merges and show trailers only on direct-push branches) to
  quantify just how censored this signal is.

## 4. Outcome metrics (never SLOC alone)

Per repo × time-bucket, per developer where possible:

Volume / velocity:
- `insertions`, `deletions`, `churn = ins+del`, `net = ins-del`.
- `active_days`, `commit_count`, `distinct_authors`.

Density / quality (guards against "it's just bloat"):
- `comment_to_code_ratio` (via a lightweight tokenizer, e.g. `pygments`/`lizard`).
- `mean_cyclomatic_complexity` per changed function (`lizard`, multi-language).
- `avg_change_size` and its dispersion (AI may fragment or enlarge commits).
- `bugfix_commit_ratio` (message classifier: fix/bug/hotfix/revert).
- `revert_rate` and `churn_recycling` = lines added then deleted within N weeks
  (a rework/instability proxy — is the new volume durable?).

The headline claim is only credible as a *joint* movement: e.g. velocity up
**and** durable-churn up **without** complexity/bugfix-ratio exploding. Velocity
up with rework and complexity up is "bloat", and we must be willing to report
that outcome.

### 4.1 Productivity vs. volume (a first-class research question)

A central goal is to test whether AI yields a real **productivity** gain, not just
more lines. These are not the same, and conflating them is the classic mistake:
an LLM can emit large volumes of code that is later rewritten, duplicated, or
never needed. So we separate three distinct constructs and report them side by
side:

- **Output volume** — raw churn/SLOC. The weakest proxy; goes up trivially if AI
  just produces more text. Necessary to measure, never sufficient to claim.
- **Durable output** — lines that survive (not deleted/reverted within N months);
  `net_durable = added − (added-then-removed)`. This is closer to delivered work.
- **Throughput of valued work** — bug-fix/feature commits merged per active
  developer-month, and (where issue/PR data is available) **cycle time**
  (issue-open → merge) and merged-PR rate. This is the productivity construct that
  a manager actually cares about.

A defensible "productivity increased" conclusion therefore requires: durable
output **and** valued-work throughput rising, while rework/complexity do **not**
rise proportionally. If only raw volume rises, the honest finding is "more code,
not more productivity" — possibly even negative (Jevons-style: cheaper code → more
churn to maintain). The study is explicitly designed to be able to return that
result; that falsifiability is what makes a *positive* productivity finding mean
something.

### 4.2 Management KPI track (raw volume as "productivity")

By stakeholder request we also report the **naive management view**: raw SLOC /
churn per developer-month, treated as the productivity headline (`more code =
more productive`). This is what upper management values, and there is no point
hiding it. We compute it as a first-class output:

- `management_kpi`: gross source SLOC added per active developer-month, indexed to
  the pre-AI baseline (100 = baseline), so the headline reads "+X% output".
- Presented on its own dashboard tile / plot, deliberately simple.

The scientific track (durable output, throughput, dose-response, bloat check) sits
**beside** it, not instead of it. Every management-KPI figure carries a one-line
caveat linking to the durable/quality panel, so the naive number is available but
never stands alone in the record. This dual-track design lets the study serve both
audiences without compromising either: the KPI people get their number, the method
stays honest about what it does and does not mean.

## 5. Noise filtering, inclusion criteria, outlier policy

Answering your question about sensible thresholds directly.

### 5.1 Repo inclusion (continuity gate)

A repo enters the panel only if it can actually inform a before/after contrast:
- **Span:** ≥ 18 months of history before its anchor and ≥ 9–12 months after.
- **Continuity:** active in ≥ 70% of the months in its window, and **no gap > 3
  consecutive months** with zero commits. This kills abandoned/one-shot repos
  and seasonal dead projects that would otherwise masquerade as "changes".
- **Team size:** ≥ 3 distinct human authors. Single-dev repos are dominated by
  one person's mood and inflate variance without adding signal (exactly your
  "one dev does more one week, less the next" concern). Keep a separate single-dev
  stratum for curiosity, never in the main pool.
- **Volume floor:** ≥ ~200 commits and some minimum median weekly churn, to avoid
  dividing by near-zero baselines.
- **Not a mirror/monorepo-vendor dump:** drop repos whose history is dominated by
  vendored/generated paths (see 5.3).

### 5.2 Bot / non-human / mechanical commits

Exclude: `dependabot`, `renovate`, `github-actions`, release bots, `[bot]`
authors; **merge commits** (`--no-merges`); giant mechanical commits (formatter
runs, license headers, mass renames) via a per-commit churn cap + file-count cap.

### 5.3 Path filtering (the biggest SLOC-inflation trap)

Compute churn **only on human-authored source**. Exclude via pathspec + a
`linguist`-style rule set: lockfiles (`*.lock`, `package-lock.json`), `vendor/`,
`third_party/`, `node_modules/`, minified assets, generated code (`*.pb.go`,
`*_pb2.py`, generated OpenAPI), migrations, test fixtures/snapshots, binaries.
Without this, "AI velocity" is mostly `pnpm-lock.yaml`.

### 5.4 Outlier handling (statistics, not deletion)

Do not hand-remove "weird" repos — that is p-hacking. Instead:
- **Log-transform** churn (heavy right tail; `log1p`).
- **Winsorize** per-repo weekly metrics at the 1st/99th percentile.
- **Robust central tendency:** report median + MAD alongside means.
- **Down-weight giants:** mixed-effects model with repo random effects, or
  inverse-variance / capped weights, so 3 mega-repos don't drive the result.
- Pre-register these rules *before looking at outcomes*.

### 5.5 Why single repos won't show it and the aggregate will

This is a signal-to-noise fact, and it is a feature, not a bug. Per-repo weekly
churn has enormous idiosyncratic variance (releases, refactors, vacations). The
AI effect, if real, is a small shift of the *mean* buried under that variance.
Pooling N repos in a hierarchical model shrinks the standard error ~1/sqrt(N):
the effect becomes visible in aggregate precisely while individual repos stay
noisy. The correct visualization is therefore a **meta-analytic forest plot /
pooled event-study**, not N separate line charts. We should say this explicitly
in the paper so a null in single repos is not misread as "no effect".

## 6. Statistical model (concrete)

Panel: repo × month (and developer × month for Design C).

- **Primary 1 (Design A):** per-repo counterfactual model (log-linear or NB trend
  + month-of-year seasonality + repo-age spline, fit on ≤2021), forward-projected;
  pooled excess = actual − expected, aggregated with repo random effects. Bayesian
  structural time-series (CausalImpact-style) as the fully-probabilistic variant.
- **Primary 2 (Design B):** reduced-form event study on the availability shock —
  aggregate and per-language `effect(t)` relative to the calendar cutoff, with a
  placebo-date null band. This is ITT: "effect of AI becoming available".
- **Backbone regression:** hierarchical / mixed-effects
  `log(churn)_{i,t} ~ post_ai * ai_suitability + repo_age_spline + team_size
  + author_tenure + (1 | repo) + (1 | primary_language) + (1 | author)`.
  The `post_ai × ai_suitability` interaction *is* the dose-response test (2.3);
  the repo-age spline absorbs lifecycle (early feature-velocity vs. late
  stabilization — your lifecycle-variance concern).
- **Validation only (Design D):** Callaway–Sant'Anna staggered DiD on the
  positive-only signature subsample, reported as corroboration with its selection
  bias stated, never as the population estimate.
- Inference: cluster-robust SEs by repo; Benjamini–Hochberg across the metric
  family (many outcomes → control the false-discovery rate).
- Effect size in **plain units**: "+X% monthly source churn per developer, 95% CI
  […]", plus a standardized effect, not just a p-value.

Placebo & robustness battery (all pre-registered):
- Placebo cutoffs in the pre-period (pretend AI arrived 2019) → excess must vanish.
- Negative-control outcomes (docs/config-only churn) → must not track source churn.
- Dose-response must be monotone in AI-suitability, near-zero for niche languages.
- Leave-one-language-out and leave-one-mega-repo-out.
- Design D vs. Design A: do direction/magnitude agree on the visible subsample?

## 7. Model-release timeline (event-study anchors)

Global shocks for Design A and for interpreting when signal appears. Dates near
the knowledge cutoff should be verified before publication (marked ⚠).

| Date        | Event                                   | Relevance                    |
|-------------|-----------------------------------------|------------------------------|
| 2021-06-29  | GitHub Copilot technical preview        | earliest broad code-AI       |
| 2022-06-21  | Copilot general availability            | first mass code-AI adoption  |
| 2022-11-30  | ChatGPT (GPT-3.5) launch                | dominant general inflection  |
| 2023-03-14  | GPT-4 / first public Claude             | capability jump              |
| 2023-07-11  | Claude 2                                | longer context               |
| 2023 (year) | Cursor editor gains traction            | AI-native IDE                |
| 2024-03-04  | Claude 3 (Opus/Sonnet/Haiku)            | quality step-up              |
| 2024-05-13  | GPT-4o                                   | —                            |
| 2024-06-20  | Claude 3.5 Sonnet                       | strong coding uptake         |
| 2025-02-24  | Claude Code (research preview)          | leaves `Co-Authored-By` ⭐    |
| 2025+       | Claude Opus 4.x line                     | ⚠ verify exact dates         |

The `Co-Authored-By: Claude` trailer from Claude Code (2025) is the cleanest
repo-level treatment marker available and should be a first-class signal, not a
footnote. Copilot GA (2022-06) and ChatGPT (2022-11) are the strongest aggregate
anchors for Design A.

## 8. Data pipeline (extraction architecture, not yet implemented)

Design goals: throughput via parallelism, bounded disk, no working-tree checkout.

- **No full clone.** Use `git clone --bare --filter=blob:none` (partial clone) or
  fetch history metadata only. Churn comes from `git log --no-merges --numstat
  --pretty=...`; complexity/comment metrics need blobs only for *changed* files,
  fetched lazily. Never check out a working tree — that is where disk explodes.
- **Parallelism with backpressure:** a bounded worker pool (e.g. `asyncio` +
  semaphore, or a process pool sized to cores) cloning into the scratch dir. A
  **disk-usage guard** polls free space; workers block (wait longer, as you said)
  rather than OOM/ENOSPC the box. Each repo is parsed, reduced to a compact
  per-week Parquet row set, then its clone is deleted immediately.
- **Idempotent & resumable:** per-repo checkpoint so a crash doesn't restart the
  fleet. Deterministic ordering, fixed seeds, `LC_ALL=C` for all git parsing.
- **Reduce early:** we persist aggregated time-series (repo × week × metric),
  never raw diffs. Final artifact: tidy Parquet + CSV/JSON ready for R/pandas.

Suggested layout:
```
src/  extract/   git mining, treatment detection, metric computation
      panel/     build repo×week and author×week panels, inclusion gate
      stats/     DiD (callaway–santanna), mixed-effects, placebo battery
      viz/       matplotlib, modern style, forest + event-study plots
data/ raw-cache/ (gitignored)   panels/ (parquet)   results/
```

## 9. Visualization plan (matplotlib, modern technical style)

Per the house style: no titles, hidden top/right spines, muted palette, light
background, direct annotations, error bars/bands everywhere.

- **Excess-churn plot (Design A):** actual vs. pre-AI counterfactual with shaded
  prediction band; the divergence after the shock *is* the effect. Placebo-cutoff
  version beside it must show no divergence.
- **Dose-response plot (the identifying argument):** x = language AI-suitability,
  y = estimated post-shock excess with CI. A positive slope (big for Python/JS,
  ~0 for niche languages) is the headline evidence; a flat line kills the AI claim.
- **Pooled event study:** x = months relative to the calendar shock, y = effect on
  log churn, shaded 95% band, vertical rule at t=0, flat pre-period as the
  no-pre-trend check.
- **Forest plot:** per-repo excess + CI, ordered, pooled diamond at the bottom —
  the "some repos commit way more" view, showing aggregate signal emerging from
  noisy individuals.
- **Timeline strip:** aggregate monthly churn with model-release markers (Sec. 7)
  as annotations — the intuitive "AI in play" picture, clearly labeled descriptive.
- **Density-vs-velocity quadrant:** did volume rise with or without complexity/
  rework rising — the bloat check.

## 10. Threats to validity (state them, don't hide them)

- Unobservable adoption (trailers stripped) → we cannot identify *who* used AI,
  only the availability shock; the claim is ITT ("effect of AI being available"),
  stated as such, with the signature subsample as biased corroboration only.
- No never-treated controls post-shock → the counterfactual carries the load, so
  everything hinges on the pre-AI model being right; guarded by placebo cutoffs,
  negative-control outcomes, and the AI-suitability dose-response.
- Coincident post-2022 secular trends (remote work, tooling) → only the
  dose-response gradient distinguishes these from AI; a flat gradient = no claim.
- SLOC/churn remain coarse proxies for "work"; density metrics are partial.
- Selection: OSS public repos ≠ industry; results don't generalize to private
  corp codebases without caveat.
- Simpson's paradox across languages → always stratify by primary language.

## 11. What a defensible conclusion looks like

Not "AI increased SLOC by X%." Rather: "Across N repos, monthly source churn per
active developer runs +X% (95% CI …) above the pre-AI counterfactual after AI
tools became available; the excess is **monotone in language AI-suitability**
(large for Python/JS/TS, indistinguishable from zero for niche languages), does
not appear on placebo cutoffs, is not mirrored by non-code negative-control
outcomes, and holds within the same developers — accompanied by [stable / rising]
complexity and rework, consistent with [genuine throughput gain / partial bloat]."
The dose-response gradient, not a raw before/after mean, is what makes this AI
rather than a generic post-2022 trend. The honest null is clearly reportable if
the gradient is flat or the placebos/negative controls fail.

## Locked decisions (2026-07-03)

1. **Sampling frame:** stratified sample drawn from **GH Archive** (activity- and
   age-stratified), not curated top-stars — chosen for representativeness over
   convenience. Star-count kept only as a covariate, never as the sampling axis.
2. **Time bucket:** **monthly** panels — smoother, robust to week-level noise
   (vacations, release spikes); we accept the modest power loss vs. weekly.
3. **Milestone:** build the **full statistical stack** end-to-end, but with the
   revised backbone: counterfactual excess-churn (Design A) + reduced-form event
   study (Design B) + dose-response mixed-effects, with Callaway–Sant'Anna DiD
   kept only as the Design-D validation on the signature subsample. Built
   bottom-up: extraction → panel → stats → viz, wired on a synthetic panel first
   so the whole pipeline is runnable before real GH Archive data lands.

Superseded (2026-07-03): AI-attribution signatures were originally the DiD
treatment variable; demoted to positive-only validation label because trailers
are routinely stripped (squash/rebase/hooks/opt-out). See Section 2.
