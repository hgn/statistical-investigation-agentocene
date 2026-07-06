"""Public GitHub repository source.

Discovers repos via the REST search API (stdlib urllib, no SDK). Honours the
rate limit by reading the ``X-RateLimit-*`` headers and sleeping until reset,
so discovery never hammers the API. Cloning itself is anonymous https and does
not consume API quota.

Sampling note (see concept.md sec. 1): search ranks by the given query, which
is a convenience frame, not a random sample. For the real study, feed a
stratified id list via the ``list`` source built from GH Archive; this source is
for bootstrapping and smaller runs.

Three cheap pre-clone filters (API calls only, no clone) keep obviously
unusable candidates from ever reaching the miner -- cloning/mining a repo just
to discover it can never pass the inclusion gate is the single biggest waste of
time observed in practice:

* ``created:<...`` in the default query excludes repos too young to ever supply
  the required pre-AI baseline (concept.md sec. 5.1's ``min_pre_months``).
* a coding-repo heuristic (known source language + a name/description
  blocklist) excludes proxy-list/IPTV/wordlist/package-index repos, which
  dominate raw GitHub search results and structurally cannot inform a
  source-churn study.
* a recent-activity check (>= N commits in the last window) via the commits
  API filters out repos that are technically "pushed recently" but not
  actually under real development, without needing a full clone to find out.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

from ..mining.language import _EXT_LANG
from .base import RepoRef, RepoSource

_API = "https://api.github.com"

# GitHub's linguist language names (as returned by the search API's `language`
# field) that map onto a "real source code" repo. Built from the same
# extension->language table the miner uses, plus GitHub's own capitalisation
# for names that don't match 1:1.
_CODE_LANGUAGES: set[str] = set(_EXT_LANG.values()) | {
    "c++", "c#", "objective-c", "objective-c++", "jupyter notebook",
}
_GITHUB_LANG_ALIASES: dict[str, str] = {
    "c++": "cpp", "c#": "csharp", "objective-c": "objc", "objective-c++": "objc",
}

# Name/description keywords that overwhelmingly mark non-development data-dump
# repos (proxy/IPTV/VPN lists, wordlists, subscription aggregators) rather than
# a software project with genuine source-code velocity. Heuristic, not exact:
# false positives/negatives are expected and acceptable at this filtering stage
# (the inclusion gate and manual review remain the real gate).
_BLOCKLIST_KEYWORDS: tuple[str, ...] = (
    "proxy", "proxies", "iptv", "v2ray", "vpn", "clash", "v2raycfg", "shadowsocks",
    "m3u", "playlist", "subscribe", "-sub", "wordlist", "ip-list", "iplist",
    "free-proxy", "freeproxy", "getproxy", "socks5",
)


class GitHubSource(RepoSource):
    name = "github"

    def __init__(
        self,
        query: str = "stars:>100 pushed:>2024-01-01 created:<2021-06-01",
        token: str | None = None,
        max_pages: int = 10,
        per_page: int = 100,
        base_url: str = _API,
        require_code_language: bool = True,
        min_recent_commits: int = 20,
        recent_window_days: int = 30,
    ) -> None:
        self._query = query
        self._token = token or os.environ.get("GITHUB_TOKEN")
        self._max_pages = max_pages
        self._per_page = min(per_page, 100)
        self._base = base_url.rstrip("/")
        self._require_code_language = require_code_language
        self._recent_window_days = recent_window_days

        # The recent-activity check costs one extra "core" API call per
        # candidate (the commits endpoint isn't covered by the separate search
        # quota). Unauthenticated core quota is only 60 req/hour -- nowhere near
        # enough to check even a few dozen candidates, so without a token this
        # would either stall the whole run behind an hour-long rate-limit sleep
        # or silently starve later pipeline stages. Disable it by default when
        # unauthenticated rather than doing that quietly.
        if min_recent_commits > 0 and not self._token:
            print(
                "[github] no GITHUB_TOKEN set: unauthenticated core API quota is "
                "60 req/hour, too little to sustain the recent-activity check "
                "(one call per candidate). Disabling --min-recent-commits for "
                "this run; set GITHUB_TOKEN (5000 req/hour) to enable it.",
                file=sys.stderr,
            )
            min_recent_commits = 0
        self._min_recent_commits = min_recent_commits

    # -- discovery -----------------------------------------------------------

    def iter_repos(self) -> Iterator[RepoRef]:
        for page in range(1, self._max_pages + 1):
            params = urllib.parse.urlencode(
                {
                    "q": self._query,
                    "sort": "updated",
                    "order": "desc",
                    "per_page": self._per_page,
                    "page": page,
                }
            )
            url = f"{self._base}/search/repositories?{params}"
            payload = self._get(url)
            if payload is None:
                return
            items = payload.get("items", [])
            if not items:
                return
            for it in items:
                ref = self._to_ref(it)
                if not self._looks_like_code_repo(ref):
                    continue
                if not self._has_recent_activity(ref):
                    continue
                yield ref

    def _to_ref(self, it: dict[str, object]) -> RepoRef:
        return RepoRef(
            provider=self.name,
            repo_id=str(it.get("id")),
            name=str(it.get("full_name")),
            clone_url=str(it.get("clone_url")),
            default_branch=(str(it["default_branch"]) if it.get("default_branch") else None),
            meta={
                "stars": it.get("stargazers_count"),
                "forks": it.get("forks_count"),
                "created_at": it.get("created_at"),
                "pushed_at": it.get("pushed_at"),
                "primary_language": it.get("language"),
                "size_kb": it.get("size"),
                "archived": it.get("archived"),
                "description": it.get("description"),
            },
        )

    # -- pre-clone filters -----------------------------------------------------

    def _looks_like_code_repo(self, ref: RepoRef) -> bool:
        if not self._require_code_language:
            return True
        lang = str(ref.meta.get("primary_language") or "").strip().lower()
        if not lang or lang not in _CODE_LANGUAGES:
            return False
        hay = f"{ref.name} {ref.meta.get('description') or ''}".lower()
        if any(kw in hay for kw in _BLOCKLIST_KEYWORDS):
            return False
        return True

    def _has_recent_activity(self, ref: RepoRef) -> bool:
        if self._min_recent_commits <= 0:
            return True
        since = (
            datetime.now(timezone.utc) - timedelta(days=self._recent_window_days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = urllib.parse.urlencode(
            {"since": since, "per_page": self._min_recent_commits}
        )
        branch = f"&sha={ref.default_branch}" if ref.default_branch else ""
        url = f"{self._base}/repos/{ref.name}/commits?{params}{branch}"
        items = self._get_list(url)
        if items is None:
            return True  # API hiccup (e.g. empty repo, 409): don't punish the candidate
        return len(items) >= self._min_recent_commits

    # -- auth for cloning ----------------------------------------------------

    def authorize_url(self, ref: RepoRef) -> str:
        # Anonymous clone for public repos. For higher clone throughput a token
        # can be embedded, but we keep public traffic tokenless by default.
        return ref.clone_url

    # -- rate-limited GET ----------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "aisloc-gatherer",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _get(self, url: str, retries: int = 5) -> dict[str, object] | None:
        result = self._request_json(url, retries)
        return result if isinstance(result, dict) else None

    def _get_list(self, url: str, retries: int = 5) -> list[object] | None:
        result = self._request_json(url, retries, quiet_codes=(404, 409))
        return result if isinstance(result, list) else None

    def _request_json(
        self, url: str, retries: int, quiet_codes: tuple[int, ...] = ()
    ) -> object | None:
        for attempt in range(retries):
            req = urllib.request.Request(url, headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    self._respect_rate_limit(resp.headers)
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in (403, 429):
                    self._sleep_until_reset(e.headers, attempt)
                    continue
                if e.code not in quiet_codes:
                    print(f"[github] HTTP {e.code} for {url}", file=sys.stderr)
                return None
            except (urllib.error.URLError, TimeoutError) as e:
                back = min(2**attempt, 30)
                print(f"[github] {e}; retry in {back}s", file=sys.stderr)
                time.sleep(back)
        return None

    def _respect_rate_limit(self, headers: object) -> None:
        try:
            remaining = int(headers.get("X-RateLimit-Remaining", "1"))  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return
        if remaining <= 1:
            self._sleep_until_reset(headers, 0)

    def _sleep_until_reset(self, headers: object, attempt: int) -> None:
        reset = None
        try:
            reset = int(headers.get("X-RateLimit-Reset"))  # type: ignore[union-attr,arg-type]
        except (TypeError, ValueError):
            reset = None
        if reset:
            delay = max(1.0, reset - time.time()) + 1.0
        else:
            delay = min(2**attempt, 60)
        print(f"[github] rate-limited; sleeping {delay:.0f}s", file=sys.stderr)
        time.sleep(delay)
