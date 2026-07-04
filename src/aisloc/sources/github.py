"""Public GitHub repository source.

Discovers repos via the REST search API (stdlib urllib, no SDK). Honours the
rate limit by reading the ``X-RateLimit-*`` headers and sleeping until reset,
so discovery never hammers the API. Cloning itself is anonymous https and does
not consume API quota.

Sampling note (see concept.md sec. 1): search ranks by the given query, which
is a convenience frame, not a random sample. For the real study, feed a
stratified id list via the ``list`` source built from GH Archive; this source is
for bootstrapping and smaller runs.
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

from .base import RepoRef, RepoSource

_API = "https://api.github.com"


class GitHubSource(RepoSource):
    name = "github"

    def __init__(
        self,
        query: str = "stars:>100 pushed:>2024-01-01",
        token: str | None = None,
        max_pages: int = 10,
        per_page: int = 100,
        base_url: str = _API,
    ) -> None:
        self._query = query
        self._token = token or os.environ.get("GITHUB_TOKEN")
        self._max_pages = max_pages
        self._per_page = min(per_page, 100)
        self._base = base_url.rstrip("/")

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
                yield self._to_ref(it)

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
            },
        )

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
