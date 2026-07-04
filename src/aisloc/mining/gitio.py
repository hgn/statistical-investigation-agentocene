"""Async git plumbing: bounded clone + streaming log.

Design choices that keep disk/memory bounded (concept.md sec. 8):
* ``--bare``: no working-tree checkout, only the object store.
* ``--shallow-since=<baseline>``: fetch only the history we analyse, not the
  whole repo back to its first commit. Falls back to a full bare clone if the
  server rejects shallow-since (then we bound with ``--since`` at log time).
* ``--single-branch --no-tags``: default branch only.
* a disk watchdog races the clone and kills it if it blows past the per-repo cap.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..resources import DiskWatchdog


class GitError(RuntimeError):
    pass


@dataclass
class CloneResult:
    path: Path
    shallow: bool


def _git_env(cfg: Config, source_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",  # never block on credential prompts
            "GIT_ASKPASS": "true",
            "GCM_INTERACTIVE": "never",
            "LC_ALL": "C",  # deterministic parsing (global convention)
        }
    )
    env.update(cfg.git_extra_env)
    env.update(source_env)
    return env


async def clone_bounded(
    url: str,
    dest: Path,
    cfg: Config,
    source_env: dict[str, str],
) -> CloneResult:
    """Clone ``url`` into ``dest`` (bare, history bounded to baseline).

    Raises ``GitError`` on failure or if the watchdog trips.
    """
    env = _git_env(cfg, source_env)
    base_args = ["--bare", "--single-branch", "--no-tags", "--quiet"]

    shallow_args = base_args + [f"--shallow-since={cfg.baseline_since}"]
    if await _try_clone(url, dest, shallow_args, cfg, env):
        return CloneResult(dest, shallow=True)

    # Some servers/protocols reject shallow-since. Retry as a full bare clone;
    # history is then bounded at log time via --since, and disk is still guarded
    # by the watchdog.
    _rmtree(dest)
    if await _try_clone(url, dest, base_args, cfg, env):
        return CloneResult(dest, shallow=False)
    raise GitError(f"clone failed: {url}")


async def _try_clone(
    url: str, dest: Path, args: list[str], cfg: Config, env: dict[str, str]
) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "git", "clone", *args, url, str(dest),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    tripped = asyncio.Event()
    watch = asyncio.create_task(
        DiskWatchdog(dest, cfg.limits.per_repo_disk_cap_bytes).watch(tripped)
    )
    waiter = asyncio.create_task(proc.wait())
    trip_wait = asyncio.create_task(tripped.wait())
    try:
        done, _pending = await asyncio.wait(
            {waiter, trip_wait},
            timeout=cfg.clone_timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if trip_wait in done:  # cap exceeded -> kill
            _kill(proc)
            return False
        if waiter not in done:  # timeout
            _kill(proc)
            return False
        return proc.returncode == 0
    finally:
        watch.cancel()
        for t in (waiter, trip_wait):
            if not t.done():
                t.cancel()
        if proc.returncode is None:
            _kill(proc)


def _kill(proc: asyncio.subprocess.Process) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass


async def stream_log(
    repo: Path, args: list[str], cfg: Config, source_env: dict[str, str]
) -> AsyncIterator[bytes]:
    """Yield raw stdout lines from ``git -C repo log <args>`` as they arrive, so
    large histories never materialise fully in memory."""
    env = _git_env(cfg, source_env)
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(repo), "log", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    assert proc.stdout is not None
    try:
        async for line in proc.stdout:
            yield line
        await asyncio.wait_for(proc.wait(), timeout=cfg.log_timeout_s)
    finally:
        if proc.returncode is None:
            _kill(proc)


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def cleanup(path: Path) -> None:
    """Delete a clone immediately after mining (frees disk for the next repo)."""
    _rmtree(path)
