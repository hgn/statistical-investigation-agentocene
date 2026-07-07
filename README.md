# Statistical Investigation: Agentocene

Empirical mining pipeline that tests, across many git repositories, whether the
arrival of generative-AI coding tools caused a measurable shift in how much code
teams produce, and whether that shift is real productivity or just more volume.

## The problem, and why the obvious approach fails

The tempting design is "average SLOC per week before ChatGPT vs. after". It does
not survive scrutiny, for four reasons:

1. **No control group.** After late 2022 essentially everyone is exposed, so any
   coincident trend (remote work, tooling, repo growth) is confounded with "AI".
2. **Adoption is unobservable.** Commit trailers like `Co-Authored-By: Claude`
   get stripped by squash/rebase/hooks/opt-out, so you cannot reliably tell which
   repo adopted AI or when. Their absence means nothing.
3. **SLOC is a poor productivity metric.** More lines can be bloat, generated
   code, or vendored dependencies.
4. **Survivorship.** Repos still alive in 2025 are not a random sample of 2020.

This project is built around those four problems rather than ignoring them. The
identification does not depend on observing per-repo adoption. Instead it uses:

- an **excess-churn counterfactual** ("excess mortality" style): fit each repo's
  pre-AI trajectory, project it forward, measure the residual;
- a **reduced-form event study** on the AI *availability* shock (intention-to-treat);
- a **dose-response by language AI-suitability** as the key identifying argument:
  a genuine AI effect must be large where the tools are strong (Python/JS/TS) and
  near zero where they are weak (COBOL, Fortran, Verilog). A flat gradient kills
  the claim, which is what makes a positive result meaningful;
- a **PU (positive-unlabeled) propensity model** that outputs a calibrated
  probability that each repo and each developer uses AI, since signatures give
  positives only.

The full methodology, including threats to validity, is in [`concept.md`](concept.md).

## What it produces

- `data/results/repo-manifest.{txt,csv}` — every examined repo by URL with:
  used for analysis (yes/no and why not), contributor count, calibrated
  probability of AI use, churn, languages, signatures.
- `data/panels/*` — tidy CSV/Parquet panels (repo x month x language, author x
  month) ready for R/Pandas.
- `data/results/*` — event-study, dose-response, backbone regression,
  management KPI, placebo check, per-repo and per-developer propensity, plus
  the developer-level outputs below.
- `data/results/dev-*` — the developer-level analysis: `dev-event-study.csv`,
  `dev-dose-response.csv` (each developer's own excess vs. their own P(AI)),
  `dev-before-after.csv`, `dev-placebo.json`, `dev-top-gainers.csv` (standout
  individuals), `dev-pooled-event-study.csv` (cross-repo aggregated).
- `data/results/fig-*.png` — figures (event study, dose-response, management KPI,
  propensity, before/after, and the developer-level analogues), modern
  technical style, 300 DPI.

## Quickstart

```sh
make help                 # list commands
make get-data             # gather git history (cached: no-op if data exists)
make analyse              # build panels, run stats, render figures, write manifest
```

Data is cached. `get-data` and the analysis stages skip work when their outputs
already exist, so re-running is cheap. To force a fresh fetch and recompute:

```sh
make data-clear           # the ONLY command that deletes collected data
make get-data analyse
```

Try it without any network using a synthetic dataset with a known planted effect:

```sh
make synth                # 80 synthetic repos with a built-in AI dose-response
make analyse
```

### Choosing repos to gather

```sh
# GitHub search (bootstrapping / small runs)
make get-data PROVIDER=github QUERY='stars:>100 language:python pushed:>2024-01-01' TARGET=300

# a curated / stratified list (recommended for the real study; see concept.md)
make get-data PROVIDER=list LIST=data/my-repos.txt TARGET=1000
```

### Low-level targets

`gather`, `synth`, `panel`, `stats`, `plots`, `manifest`, `test`, `typecheck`
run a single stage directly, ignoring the cache. `results-clear` drops derived
panels/results but keeps raw records.

## How it works

- **Gathering** (`aisloc.sources`, `aisloc.mining`, `aisloc.gather`): pure Python
  standard library plus the `git` binary. Repos are cloned `--bare
  --shallow-since` (history bounded to the baseline, no working tree), mined with
  streaming `git log --numstat`, reduced to a compact per-repo JSONL record, then
  the clone is deleted. Runs in parallel under a **resource governor** that watches
  disk/memory/load and throttles concurrency (waiting rather than risking ENOSPC
  or the OOM killer), with a per-repo disk watchdog. The run is resumable and shows
  live multi-line progress with an ETA.
- **Analysis** (`aisloc.analysis`): reads only the JSONL records, never git or the
  network. Consolidates panels, runs the statistical stack (numpy/scipy only),
  renders figures.

Author emails are pseudonymised (salted hash) at record time, so the same
developer is joinable across views without storing raw addresses.

```
src/aisloc/
  sources/    provider abstraction: github, gitlab (full-instance admin discovery), list
  mining/     bounded clone, churn/signature extraction, language + path filters
  analysis/   panel, stats, plots, manifest, inclusion gate, synthetic generator
  gather.py   parallel, resource-aware, resumable orchestrator
```

## Volume vs. productivity

The study separates raw output (churn), durable output (lines that survive), and
throughput of valued work (bug-fix/feature cadence, cycle time). By stakeholder
request it also reports the naive **management KPI** (raw SLOC per developer-month
as "productivity") as a first-class headline, but always beside the durability and
quality panels, never instead of them. If only raw volume rises, the honest
finding is "more code, not more productivity". See `concept.md` sec. 4.

## Developer-level analysis

AI adoption is a per-developer choice, not a repo-level one: within the same
project some contributors use it heavily, others not at all. Pooling to the
repo mean dilutes exactly that signal, and worse, it's confounded by team-size
composition changes over a project's lifetime — a repo gaining more casual
contributors mechanically lowers "churn per developer" with nothing to do with
AI. `aisloc.analysis.stats` also runs the whole excess-churn / placebo / dose-
response machinery at **individual-developer granularity**: each contributor's
own churn trajectory, before vs. after, dosed by their own PU-model `p_ai`
rather than by language suitability. Tracking the same person over time is
immune to team growth elsewhere, and empirically this shows up exactly as
expected: the developer-level placebo check comes out near zero, while the
repo-level one does not — team-size composition is a real confound at repo
level, and analysing the same individuals over time route around it.

Beyond the population-average dose-response, two further angles matter because
an AI effect is very unlikely to be uniform across people:

- **Heterogeneity, not just the mean** (`heterogeneity_by_p_ai_group` in
  `stats-summary.json`, `dev-top-gainers.csv`): a slope near zero can still
  hide a real effect concentrated in a subset of individuals. We report the
  share of developers with a large (>=50%) personal jump, dispersion (std,
  p90) by AI-likelihood group, and a leaderboard of the standout individuals —
  descriptive, not a causal claim, but the right place to look for
  effects an average would wash out.
- **Cross-repo aggregation** (`cross_repo_overlap`, `dev-pooled-event-study.csv`):
  the pseudonymised developer hash is already stable across repos (same salted
  email -> same hash), so a contributor active in several sampled repos can be
  tracked as one person across all of them. With a broad random sample this
  overlap is thin (most contributors appear in only one sampled repo); it gets
  much more powerful with a **deliberately connected sample** — one ecosystem's
  core maintainers, who by construction work across many of that ecosystem's
  repos. `data/example-rust-core-repos.txt` is a ready-to-use seed list for
  this (`make get-data PROVIDER=list LIST=data/example-rust-core-repos.txt`);
  swap in whichever ecosystem is of interest, the property that matters is
  "many repos, shared core contributors", not Rust specifically.

A methodological lesson worth keeping visible: an earlier attempt to fix
lifecycle curvature (accelerating/decelerating growth) by adding a quadratic
trend term to the per-entity counterfactual fit made *both* placebo checks
worse, not better — naive polynomial extrapolation over a 24-30 month post
window is numerically unstable. It was reverted in favour of a linear trend,
with lifecycle handled only where it can be identified safely: as an explicit
repo-age covariate in the *pooled* cross-repo backbone regression, where age
genuinely varies independently of calendar time (unlike in a single entity's
own series, where age and calendar time are the same straight line up to a
shift). The placebo check is what caught this — trust it over any single
"significant" result that shows up without one.

## Volume vs. productivity

The study separates raw output (churn), durable output (lines that survive), and
throughput of valued work (bug-fix/feature cadence, cycle time). By stakeholder
request it also reports the naive **management KPI** (raw SLOC per developer-month
as "productivity") as a first-class headline, but always beside the durability and
quality panels, never instead of them. If only raw volume rises, the honest
finding is "more code, not more productivity". See `concept.md` sec. 4.

## Requirements

- Python 3.12+ and `git`.
- Gathering: no third-party packages (standard library only).
- Analysis: `pandas`, `numpy`, `scipy`, `matplotlib`. `pyarrow` is optional (adds
  Parquet output; CSV is always written). `statsmodels`/`sklearn` are **not**
  required (the stats are implemented directly).

## Status

Research tooling. The analysis code is validated end-to-end against a synthetic
dataset with a known effect (`make synth`). Model-release dates near the knowledge
horizon should be verified before publication (see `concept.md` sec. 7). Results
from public OSS repos do not automatically generalise to private/industry code.

## On-prem GitLab: full-instance access (company-internal analyses)

`GitLabSource` (`src/aisloc/sources/gitlab.py`) discovers **every project on the
GitLab instance**, not just the ones your own account is a member of, so the
study covers the whole company rather than one employee's visible slice. It
paginates `GET /api/v4/projects`, clones over HTTPS with the token embedded
(`https://oauth2:<token>@host/group/repo.git`), excludes archived projects by
default, and honours rate-limit headers with the same backoff behaviour as the
GitHub source. Nothing else changes: the resource governor, miner, panels,
statistics and figures are provider-agnostic.

### Why you need an admin token, and where to get one

GitLab's project-listing endpoint returns **only what the calling token's
account can see** -- for an ordinary account that's exactly what you'd see
browsing the UI logged in as yourself. To get *every* repo, including ones
you're not a member of and ones marked private, the token must belong to an
account with **GitLab Administrator** rights on that instance (self-managed
GitLab only; gitlab.com has no such affordance and cannot be used this way).
This is a property of GitLab's permission model, not of this code: with an
admin account, `GET /api/v4/projects` returns the entire instance; with a
regular account, it returns the regular account's own projects.

Two ways to get the token, in order of preference:

1. **Best practice -- a dedicated service account.** Ask whoever administers
   the GitLab instance to create a bot/service account, grant it
   **Administrator** rights, and issue a Personal Access Token from that
   account (`Edit profile > Access Tokens`) with scopes `read_api` and
   `read_repository`. This keeps the token auditable and revocable
   independently of any real person's account, and is what you'd want anyway
   for something that clones the whole company's history.
2. **Using an existing admin's own account.** If there's no appetite for a
   service account yet, an instance administrator can generate a Personal
   Access Token from their own `Edit profile > Access Tokens` page with the
   same scopes (`read_api`, `read_repository`). Functionally identical, just
   tied to a person rather than a bot identity.

Either way: set an expiry appropriate for how long the study runs, and store
the token only as an environment variable -- never in a file that could be
committed (see the security note below).

### Running it

```sh
export GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx   # in YOUR shell, never pasted into chat/logs
make get-data PROVIDER=gitlab BASE_URL=https://gitlab.corp.example TARGET=1000
```

`BASE_URL` isn't a Makefile variable by default (unlike `QUERY`/`TARGET`); pass
it straight through to the CLI if you're not going through `make`:

```sh
PYTHONPATH=src python3 -m aisloc.gather --provider gitlab \
    --base-url https://gitlab.corp.example --target 1000
```

Useful flags (all optional):

| Flag | Effect |
|---|---|
| `--group <path>` | restrict to one top-level group (with subgroups) instead of the whole instance |
| `--include-archived` | also discover archived projects (excluded by default) |
| `--gitlab-no-verify-ssl` | skip TLS verification, for an internal host with a self-signed cert |

Everything downstream stays local: clones are transient and deleted immediately
after mining, only pseudonymised aggregates are written to `data/`, and the
analysis half never makes a network call. Set `AISLOC_SALT` (env var) to a
private, company-specific value before gathering, so the salted developer
hashes in the output can't be correlated with anyone's identity outside the
company, and aren't reproducible even if this repo's code is later made public.

### Handling the token safely

- Never paste a token into a chat/assistant session, ticket, Slack message, or
  commit -- treat anything typed into a logged channel as compromised the
  moment it's sent, token scope notwithstanding.
- Set it only via `export GITLAB_TOKEN=...` in your own shell (or a secrets
  manager / CI secret store); it is read from the environment
  (`os.environ.get("GITLAB_TOKEN")`) and is never written to disk, logged, or
  embedded in any output file -- `data/records/*.jsonl` stores the pseudonymised
  developer hash and repo metadata only.
- If a token is ever exposed (chat, screen share, committed by accident),
  revoke it immediately in GitLab (`Edit profile > Access Tokens > Revoke`) and
  issue a new one.
