# AI-Induced SLOC Velocity

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
  management KPI, placebo check, per-repo and per-developer propensity.
- `data/results/fig-*.png` — figures (event study, dose-response, management KPI,
  propensity), modern technical style, 300 DPI.

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
  sources/    provider abstraction: github, gitlab (stub), list
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

## Switching to on-prem GitLab (company-internal analyses)

The pipeline is deliberately provider-agnostic. Moving from public GitHub to an
internal GitLab needs changes in exactly **one file** plus a config flag; the
governor, miner, panels, statistics and figures are untouched.

To do it:

1. Finish `src/aisloc/sources/gitlab.py` (`GitLabSource.iter_repos`) against the
   GitLab REST API (`GET {base_url}/api/v4/projects`, paginated via the
   `X-Next-Page` header), mapping each project to a `RepoRef`. A stub with the
   exact endpoints is already in place.
2. Provide a token via the `GITLAB_TOKEN` environment variable. Cloning private
   repos is already wired through `authorize_url`
   (`https://oauth2:<token>@host/group/repo.git`).
3. Run it:

   ```sh
   export GITLAB_TOKEN=...
   make get-data PROVIDER=gitlab   # plus base_url/group via provider options
   ```

Everything stays local: clones are transient and deleted after mining, only the
pseudonymised aggregates are written, and the analysis half never phones home,
which matters for internal codebases. Set `AISLOC_SALT` to a private value so the
developer hashes are not reproducible outside the company.

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
