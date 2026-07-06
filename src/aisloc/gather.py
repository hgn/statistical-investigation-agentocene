"""Gathering orchestrator: parallel, resource-aware, resumable.

Runs many per-repo miners concurrently under the resource governor, streams each
result to a JSONL shard, and can resume across runs (already-gathered repos are
skipped). This is the executable entry point:

    PYTHONPATH=src python -m aisloc.gather --provider github --target 500

Output: data/records/records-<runid>.jsonl (one JSON object per repo) plus
failures-<runid>.jsonl. The analysis layer consumes only these files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from .config import GiB, Config, ResourceLimits
from .mining import mine_repo
from .mining.gitio import GitError
from .progress import ProgressPrinter, fmt_dur
from .resources import ResourceGovernor
from .sources import RepoRef, make_source

_SENTINEL: object = object()
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(ref: RepoRef) -> str:
    return _UNSAFE.sub("-", f"{ref.provider}-{ref.repo_id}")[:120]


class ShardWriter:
    """Append JSONL records, rotating every ``shard_size`` records. One writer,
    guarded by an async lock; flushes each line so a crash loses at most the
    in-flight record."""

    def __init__(self, out_dir: Path, run_id: str, shard_size: int) -> None:
        self._dir = out_dir
        self._run = run_id
        self._size = shard_size
        self._n = 0
        self._idx = 0
        self._fh = None
        self._lock = asyncio.Lock()

    def _rotate(self) -> None:
        if self._fh is not None:
            self._fh.close()
        path = self._dir / f"records-{self._run}-{self._idx:03d}.jsonl"
        self._fh = path.open("a", encoding="utf-8")
        self._idx += 1

    async def write(self, obj: dict[str, object]) -> None:
        async with self._lock:
            if self._fh is None or self._n % self._size == 0:
                self._rotate()
            assert self._fh is not None
            self._fh.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._fh.flush()
            self._n += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def _load_done(records_dir: Path) -> set[str]:
    """Repo ids already gathered in any prior run (for resume).

    Keyed by (provider, name) rather than (provider, repo_id): different
    sources for the same provider tag assign repo_id differently -- ListSource
    uses the "owner/repo" name itself (no API call to resolve a real id),
    GitHubSource uses GitHub's numeric id -- so the same physical repo found
    via both a curated list and a GitHub search would silently bypass
    dedup/resume and get mined twice under repo_id keying. "name" (owner/repo)
    is the one identifier both sources agree on for the same repo.
    """
    done: set[str] = set()
    for path in records_dir.glob("records-*.jsonl"):
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = f"{obj.get('provider')}:{obj.get('name')}"
                    done.add(key)
        except OSError:
            continue
    return done


class Orchestrator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cfg.scratch_dir.mkdir(parents=True, exist_ok=True)
        cfg.records_dir.mkdir(parents=True, exist_ok=True)
        self.gov = ResourceGovernor(cfg.limits, cfg.scratch_dir)
        self.run_id = time.strftime("%Y%m%dT%H%M%S")
        self.records = ShardWriter(cfg.records_dir, self.run_id, cfg.shard_size)
        self.fail_fh = (cfg.records_dir / f"failures-{self.run_id}.jsonl").open(
            "a", encoding="utf-8"
        )
        self.done = _load_done(cfg.records_dir)
        self.ok = 0
        self.failed = 0
        self.skipped = 0
        self.inflight: set[str] = set()
        self.stop = asyncio.Event()
        self.started = time.monotonic()
        self._count_lock = asyncio.Lock()

    # -- producer/consumer ---------------------------------------------------

    async def run(self) -> int:
        self.cfg.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.records_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[gather] run={self.run_id} provider={self.cfg.provider} "
            f"target={self.cfg.target_repos} resume-known={len(self.done)}",
            file=sys.stderr,
        )
        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=self.cfg.limits.max_concurrency * 2)
        workers = [
            asyncio.create_task(self._worker(queue))
            for _ in range(self.cfg.limits.max_concurrency)
        ]
        producer = asyncio.create_task(self._produce(queue))
        reporter = asyncio.create_task(self._report())

        await producer
        for _ in workers:
            await queue.put(_SENTINEL)
        await asyncio.gather(*workers)
        reporter.cancel()
        try:
            await reporter
        except asyncio.CancelledError:
            pass

        self.records.close()
        self.fail_fh.close()
        print(
            f"[gather] done: ok={self.ok} failed={self.failed} skipped={self.skipped} "
            f"in {time.monotonic() - self.started:.0f}s",
            file=sys.stderr,
        )
        return 0 if self.ok else 1

    async def _produce(self, queue: asyncio.Queue[object]) -> None:
        source = make_source(self.cfg)
        it = iter(source.iter_repos())

        def _next() -> object:
            try:
                return next(it)
            except StopIteration:
                return _SENTINEL

        while not self.stop.is_set():
            if self._time_exhausted():
                self.stop.set()
                break
            ref = await asyncio.to_thread(_next)  # source I/O off the event loop
            if ref is _SENTINEL:
                break
            assert isinstance(ref, RepoRef)
            key = f"{ref.provider}:{ref.name}"  # see _load_done for why not repo_id
            if key in self.done:
                self.skipped += 1
                continue
            self.done.add(key)
            await queue.put(ref)

    async def _worker(self, queue: asyncio.Queue[object]) -> None:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                return
            if self.stop.is_set():
                continue  # drain quickly
            assert isinstance(item, RepoRef)
            await self._process(item)

    async def _process(self, ref: RepoRef) -> None:
        dest = self.cfg.scratch_dir / _safe_name(ref)
        self.inflight.add(ref.name)
        try:
            async with self.gov.slot():
                source = make_source(self.cfg)  # cheap; carries auth for cloning
                record = await mine_repo(ref, self.cfg, source, dest)
        except GitError as e:
            await self._note_failure(ref, f"git: {e}")
            return
        except Exception as e:  # noqa: BLE001 - one bad repo must not kill the run
            await self._note_failure(ref, f"{type(e).__name__}: {e}")
            return
        finally:
            self.inflight.discard(ref.name)

        if not record["months"]:  # nothing in the baseline window -> not useful
            await self._note_failure(ref, "empty: no source churn in window")
            return

        await self.records.write(record)
        async with self._count_lock:
            self.ok += 1
            if self.ok >= self.cfg.target_repos:
                self.stop.set()

    async def _note_failure(self, ref: RepoRef, reason: str) -> None:
        async with self._count_lock:
            self.failed += 1
        self.fail_fh.write(
            json.dumps({"provider": ref.provider, "repo_id": ref.repo_id,
                        "name": ref.name, "reason": reason}) + "\n"
        )
        self.fail_fh.flush()

    def _time_exhausted(self) -> bool:
        return (
            self.cfg.time_budget_s is not None
            and time.monotonic() - self.started > self.cfg.time_budget_s
        )

    async def _report(self) -> None:
        printer = ProgressPrinter(sys.stderr)
        try:
            while True:
                await asyncio.sleep(0.5 if printer.tty else 15)
                printer.render(self._status_lines())
        except asyncio.CancelledError:
            printer.render(self._status_lines())
            printer.finalize()
            raise

    def _status_lines(self) -> list[str]:
        elapsed = time.monotonic() - self.started
        target = self.cfg.target_repos
        done = self.ok
        rate = done / elapsed if elapsed > 0 and done else 0.0
        eta = ((target - done) / rate) if rate > 0 and done < target else None
        snap = self.gov.snapshot()

        names = sorted(self.inflight)
        shown = ", ".join(names[:4])
        if len(names) > 4:
            shown += f" (+{len(names) - 4})"

        pct = (done / target * 100) if target else 0.0
        bar_w = 24
        fill = int(bar_w * min(1.0, done / target)) if target else 0
        bar = "#" * fill + "." * (bar_w - fill)
        return [
            f"gathering repos  [{bar}] {done}/{target} ({pct:4.1f}%)",
            f"  processed {done + self.failed}  ok {done}  failed {self.failed}  "
            f"skipped {self.skipped}",
            f"  elapsed {fmt_dur(elapsed)}  eta {fmt_dur(eta)}  "
            f"rate {rate * 60:.1f}/min",
            f"  resources  concurrency<={snap.allowed}  disk {snap.free_disk >> 30}GiB  "
            f"mem {snap.free_mem >> 30}GiB  load {snap.load:.1f}  [{snap.reason}]",
            f"  in-flight  {shown or '-'}",
        ]


# -- CLI ---------------------------------------------------------------------


def _build_config(a: argparse.Namespace) -> Config:
    opts: dict[str, str] = {}
    if a.query:
        opts["query"] = a.query
    if a.token:
        opts["token"] = a.token
    if a.base_url:
        opts["base_url"] = a.base_url
    if a.group:
        opts["group"] = a.group
    if a.list:
        opts["path"] = a.list
    if a.max_pages:
        opts["max_pages"] = str(a.max_pages)
    if a.no_code_language_filter:
        opts["require_code_language"] = "false"
    opts["min_recent_commits"] = str(a.min_recent_commits)
    opts["recent_window_days"] = str(a.recent_window_days)
    if a.include_archived:
        opts["include_archived"] = "true"
    if a.gitlab_no_verify_ssl:
        opts["verify_ssl"] = "false"

    limits = ResourceLimits(
        max_concurrency=a.max_concurrency,
        min_free_disk_bytes=int(a.min_free_disk_gb * GiB),
        per_repo_disk_cap_bytes=(None if a.per_repo_cap_gb <= 0 else int(a.per_repo_cap_gb * GiB)),
    )
    return Config(
        provider=a.provider,
        provider_opts=opts,
        target_repos=a.target,
        baseline_since=a.since,
        time_budget_s=(a.time_budget * 60 if a.time_budget else None),
        scratch_dir=Path(a.scratch),
        records_dir=Path(a.records),
        shard_size=a.shard_size,
        limits=limits,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="aisloc.gather", description="Parallel, resource-aware git history gatherer."
    )
    p.add_argument("--provider", default="github", choices=["github", "gitlab", "list"])
    p.add_argument("--query", help="GitHub search query (github provider)")
    p.add_argument("--list", help="path to repo list (list provider)")
    p.add_argument("--token", help="API token (or use GITHUB_TOKEN/GITLAB_TOKEN)")
    p.add_argument("--base-url", dest="base_url", help="GitLab base URL, e.g. https://gitlab.corp.example")
    p.add_argument("--group", help="GitLab group filter (default: entire instance)")
    p.add_argument(
        "--include-archived", action="store_true",
        help="include archived projects (gitlab provider; excluded by default)",
    )
    p.add_argument(
        "--gitlab-no-verify-ssl", action="store_true",
        help="skip TLS verification (gitlab provider; only for a trusted internal "
        "host with a self-signed certificate)",
    )
    p.add_argument("--max-pages", type=int, default=10, help="GitHub search pages")
    p.add_argument(
        "--no-code-language-filter", action="store_true",
        help="disable the primary-language/keyword coding-repo heuristic (github provider)",
    )
    p.add_argument(
        "--min-recent-commits", type=int, default=20,
        help="require at least N commits in --recent-window-days before cloning "
        "(github provider; 0 disables)",
    )
    p.add_argument(
        "--recent-window-days", type=int, default=30,
        help="window for --min-recent-commits (github provider)",
    )
    p.add_argument("--target", type=int, default=500, help="stop after N gathered repos")
    p.add_argument("--since", default="2019-01-01", help="baseline start (history floor)")
    p.add_argument("--scratch", default="data/raw-cache", help="clone scratch dir")
    p.add_argument("--records", default="data/records", help="JSONL output dir")
    p.add_argument("--shard-size", type=int, default=200, help="records per shard file")
    p.add_argument("--max-concurrency", type=int, default=ResourceLimits().max_concurrency)
    p.add_argument("--min-free-disk-gb", type=float, default=5.0)
    p.add_argument("--per-repo-cap-gb", type=float, default=3.0, help="0 disables the cap")
    p.add_argument("--time-budget", type=float, default=0.0, help="wall-clock cap (minutes)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _build_config(args)
    try:
        return asyncio.run(Orchestrator(cfg).run())
    except KeyboardInterrupt:
        print("\n[gather] interrupted; partial records are on disk", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
