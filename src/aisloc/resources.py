"""Dynamic resource governor.

Pure stdlib, Linux-targeted. It watches free disk, available memory and load
average, and throttles how many repos are cloned/mined concurrently. Under
pressure it sheds concurrency down to ``min_concurrency`` and, if still tight,
blocks new work entirely ("in doubt, wait longer") instead of risking ENOSPC or
the OOM killer. A separate watchdog aborts a single clone that grows past the
per-repo disk cap.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import ResourceLimits


def _mem_available_bytes() -> int:
    """Available memory from /proc/meminfo (Linux). Falls back to a large value
    so a non-Linux dev box does not deadlock the pool."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 1 << 62


def _loadavg_1m() -> float:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return 0.0


@dataclass
class Snapshot:
    free_disk: int
    free_disk_ratio: float
    free_mem: int
    load: float
    allowed: int
    reason: str


class ResourceGovernor:
    """Async gate around per-repo work.

    Usage::

        gov = ResourceGovernor(limits, scratch_dir)
        async with gov.slot():
            ...  # clone + mine one repo
    """

    def __init__(self, limits: ResourceLimits, scratch_dir: Path) -> None:
        self._lim = limits
        self._scratch = scratch_dir
        self._in_flight = 0
        self._cond = asyncio.Condition()
        self._backoff = limits.backoff_min_s
        self._cpus = max(1, os.cpu_count() or 1)

    # -- metrics -------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        usage = shutil.disk_usage(self._scratch)
        free_ratio = usage.free / usage.total if usage.total else 1.0
        free_mem = _mem_available_bytes()
        load = _loadavg_1m()

        allowed = self._lim.max_concurrency
        reason = "ok"
        if free_mem < self._lim.min_free_mem_bytes:
            allowed = self._lim.min_concurrency
            reason = "low-mem"
        # Load-based shedding scales concurrency down smoothly.
        load_cap = int(self._cpus * self._lim.max_load_per_cpu)
        if load > load_cap and load_cap > 0:
            over = load / load_cap
            allowed = max(self._lim.min_concurrency, int(allowed / over))
            reason = "high-load" if reason == "ok" else reason
        # Disk is a hard gate: below the bound, permit nothing new.
        disk_tight = (
            usage.free < self._lim.min_free_disk_bytes
            or free_ratio < self._lim.min_free_disk_ratio
        )
        if disk_tight:
            allowed = 0
            reason = "low-disk"
        return Snapshot(usage.free, free_ratio, free_mem, load, allowed, reason)

    # -- gating --------------------------------------------------------------

    def slot(self) -> "_Slot":
        return _Slot(self)

    async def _acquire(self) -> None:
        async with self._cond:
            while True:
                snap = self.snapshot()
                if self._in_flight < snap.allowed:
                    self._in_flight += 1
                    self._backoff = self._lim.backoff_min_s
                    return
                # Constrained: log once per wait and back off (bounded, growing).
                wait = self._backoff
                self._log_pressure(snap, wait)
                self._backoff = min(self._backoff * 1.7, self._lim.backoff_max_s)
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass  # re-evaluate metrics

    async def _release(self) -> None:
        async with self._cond:
            self._in_flight -= 1
            self._cond.notify_all()

    def _log_pressure(self, snap: Snapshot, wait: float) -> None:
        print(
            f"[governor] throttling ({snap.reason}): in_flight={self._in_flight} "
            f"allowed={snap.allowed} free_disk={snap.free_disk >> 30}GiB "
            f"free_mem={snap.free_mem >> 30}GiB load={snap.load:.1f} "
            f"-> wait {wait:.0f}s",
            file=sys.stderr,
        )


class _Slot:
    def __init__(self, gov: ResourceGovernor) -> None:
        self._gov = gov

    async def __aenter__(self) -> "_Slot":
        await self._gov._acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._gov._release()


class DiskWatchdog:
    """Polls a directory's size and reports when it exceeds a cap, so the caller
    can kill a runaway clone. Cheap: uses os.walk on the (small) clone dir."""

    def __init__(self, path: Path, cap_bytes: int | None, poll_s: float = 2.0) -> None:
        self._path = path
        self._cap = cap_bytes
        self._poll = poll_s

    async def watch(self, on_exceed: "asyncio.Event") -> None:
        if self._cap is None:
            return
        while True:
            if _dir_size(self._path) > self._cap:
                on_exceed.set()
                return
            await asyncio.sleep(self._poll)


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def now() -> float:
    return time.monotonic()
