"""Central configuration for the gathering layer.

Everything tunable lives here as a frozen dataclass with sane defaults; the CLI
in ``aisloc.gather`` overrides fields from argparse/env. No magic constants are
scattered through the mining code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# Baseline start: we only mine history back to here, never the full repo. Three
# years before the ChatGPT inflection gives a solid pre-AI baseline while keeping
# clones small. See concept.md sec. 5/7.
DEFAULT_BASELINE_SINCE = "2019-01-01"

GiB = 1024**3


@dataclass(frozen=True)
class ResourceLimits:
    """Thresholds the governor enforces. All sizes in bytes."""

    max_concurrency: int = max(2, (os.cpu_count() or 4))
    min_concurrency: int = 1
    # Pause dispatching new repos when free disk drops below either bound.
    min_free_disk_bytes: int = 5 * GiB
    min_free_disk_ratio: float = 0.10
    # Pause when available memory drops below this.
    min_free_mem_bytes: int = 1 * GiB
    # Shed concurrency when 1-min load exceeds this multiple of the CPU count.
    max_load_per_cpu: float = 2.0
    # Abort a single clone whose on-disk size exceeds this (protects the box
    # from one pathological mega-repo). None disables the watchdog.
    per_repo_disk_cap_bytes: int | None = 3 * GiB
    # Backoff bounds (seconds) when resources are tight ("in doubt, wait longer").
    backoff_min_s: float = 1.0
    backoff_max_s: float = 60.0


@dataclass(frozen=True)
class Config:
    # --- provider selection -------------------------------------------------
    provider: str = "github"  # "github" | "gitlab" | "list"
    # Free-form provider options (token, base_url, sample query, seed repo list).
    provider_opts: dict[str, str] = field(default_factory=dict)

    # --- collection scope ---------------------------------------------------
    target_repos: int = 500  # stop after this many successfully gathered
    baseline_since: str = DEFAULT_BASELINE_SINCE
    time_budget_s: float | None = None  # wall-clock cap for the whole run

    # --- filesystem ---------------------------------------------------------
    scratch_dir: Path = Path("data/raw-cache")  # clones land here, then deleted
    records_dir: Path = Path("data/records")  # JSONL output shards
    shard_size: int = 200  # repos per JSONL shard file

    # --- limits -------------------------------------------------------------
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    # --- git ----------------------------------------------------------------
    clone_timeout_s: float = 900.0
    log_timeout_s: float = 600.0
    git_extra_env: dict[str, str] = field(default_factory=dict)

    def with_opts(self, **kw: object) -> "Config":
        return replace(self, **kw)  # type: ignore[arg-type]
