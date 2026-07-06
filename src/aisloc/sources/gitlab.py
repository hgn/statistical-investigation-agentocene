"""On-prem GitLab repository source: full-instance discovery for the company
deployment.

Uses the GitLab REST API's project listing endpoint. On self-managed GitLab,
``GET /api/v4/projects`` returns **every project on the instance** (public,
internal, and private, regardless of membership) when the calling token
belongs to an instance **Administrator**. A non-admin token only ever sees what
that user is a member of, same as browsing the UI logged in as an ordinary
employee -- so "see everything" is a property of the *token's account*, not of
this code. See the README's "On-prem GitLab" section for how to obtain one.

Cloning uses the same token embedded in the HTTPS URL
(``https://oauth2:<token>@host/group/repo.git``); an admin token can clone any
project it can list, including ones the admin is not an explicit member of.

Nothing else in the pipeline changes: the miner, governor, orchestrator and
output schema are all provider-agnostic.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

from .base import RepoRef, RepoSource

_INSECURE_WARNED = False


class GitLabSource(RepoSource):
    name = "gitlab"

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        group: str | None = None,
        per_page: int = 100,
        max_pages: int = 200,
        include_archived: bool = False,
        verify_ssl: bool = True,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token or os.environ.get("GITLAB_TOKEN")
        self._group = group
        self._per_page = min(per_page, 100)
        self._max_pages = max_pages
        self._include_archived = include_archived
        self._verify_ssl = verify_ssl
        if not self._token:
            print(
                "[gitlab] no token set: GET /api/v4/projects will only return "
                "public projects, same as an anonymous visitor. Set GITLAB_TOKEN "
                "to an admin token to discover private/internal projects too.",
                file=sys.stderr,
            )
        if not verify_ssl:
            self._warn_insecure()

    def _warn_insecure(self) -> None:
        global _INSECURE_WARNED
        if not _INSECURE_WARNED:
            print(
                "[gitlab] SSL verification disabled (--gitlab-no-verify-ssl): "
                "only use this against a trusted internal host with a "
                "self-signed certificate.",
                file=sys.stderr,
            )
            _INSECURE_WARNED = True

    # -- discovery -------------------------------------------------------

    def iter_repos(self) -> Iterator[RepoRef]:
        if self._group:
            base_path = f"/api/v4/groups/{urllib.parse.quote(self._group, safe='')}/projects"
            base_params = {"include_subgroups": "true"}
        else:
            # Admin-wide listing: no `membership`/`owned` filter, so an admin
            # token gets every project on the instance, not just their own.
            base_path = "/api/v4/projects"
            base_params = {"order_by": "id", "sort": "asc", "statistics": "false"}
        if not self._include_archived:
            base_params["archived"] = "false"

        page = 1
        while page <= self._max_pages:
            params = dict(base_params, per_page=str(self._per_page), page=str(page))
            url = f"{self._base}{base_path}?{urllib.parse.urlencode(params)}"
            items, next_page = self._get_page(url)
            if items is None:
                return
            for it in items:
                yield self._to_ref(it)
            if not next_page:
                return
            page = next_page

    def _to_ref(self, it: dict[str, object]) -> RepoRef:
        return RepoRef(
            provider=self.name,
            repo_id=str(it.get("id")),
            name=str(it.get("path_with_namespace")),
            clone_url=str(it.get("http_url_to_repo")),
            default_branch=(str(it["default_branch"]) if it.get("default_branch") else None),
            meta={
                "stars": it.get("star_count"),
                "forks": it.get("forks_count"),
                "created_at": it.get("created_at"),
                "pushed_at": it.get("last_activity_at"),
                "primary_language": None,  # not in the list payload; see languages() below
                "size_kb": None,
                "archived": it.get("archived"),
                "visibility": it.get("visibility"),
                "description": it.get("description"),
            },
        )

    # -- auth for cloning --------------------------------------------------

    def authorize_url(self, ref: RepoRef) -> str:
        if not self._token:
            return ref.clone_url
        parts = urllib.parse.urlsplit(ref.clone_url)
        netloc = f"oauth2:{self._token}@{parts.hostname}"
        if parts.port:
            netloc += f":{parts.port}"
        return urllib.parse.urlunsplit(
            (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
        )

    # -- rate-limited GET ----------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json", "User-Agent": "aisloc-gatherer"}
        if self._token:
            h["PRIVATE-TOKEN"] = self._token
        return h

    def _ssl_context(self) -> ssl.SSLContext | None:
        if self._verify_ssl:
            return None
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _get_page(
        self, url: str, retries: int = 5
    ) -> tuple[list[dict[str, object]] | None, int | None]:
        for attempt in range(retries):
            req = urllib.request.Request(url, headers=self._headers())
            try:
                with urllib.request.urlopen(
                    req, timeout=30, context=self._ssl_context()
                ) as resp:
                    self._respect_rate_limit(resp.headers)
                    body = json.loads(resp.read().decode("utf-8"))
                    next_page_hdr = resp.headers.get("X-Next-Page", "")
                    next_page = int(next_page_hdr) if next_page_hdr.strip() else None
                    return (body if isinstance(body, list) else None), next_page
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    self._sleep_backoff(e.headers, attempt)
                    continue
                if e.code in (401, 403):
                    print(
                        f"[gitlab] HTTP {e.code} for {url} -- check GITLAB_TOKEN "
                        "has read_api/read_repository scope (and admin rights, "
                        "for full-instance visibility)",
                        file=sys.stderr,
                    )
                    return None, None
                print(f"[gitlab] HTTP {e.code} for {url}", file=sys.stderr)
                return None, None
            except (urllib.error.URLError, TimeoutError) as e:
                back = min(2**attempt, 30)
                print(f"[gitlab] {e}; retry in {back}s", file=sys.stderr)
                time.sleep(back)
        return None, None

    def _respect_rate_limit(self, headers: object) -> None:
        try:
            remaining = int(headers.get("RateLimit-Remaining", "1"))  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return
        if remaining <= 1:
            self._sleep_backoff(headers, 0)

    def _sleep_backoff(self, headers: object, attempt: int) -> None:
        reset = None
        try:
            reset = int(headers.get("RateLimit-Reset"))  # type: ignore[union-attr,arg-type]
        except (TypeError, ValueError):
            reset = None
        delay = max(1.0, reset - time.time()) + 1.0 if reset else min(2**attempt, 30)
        print(f"[gitlab] rate-limited; sleeping {delay:.0f}s", file=sys.stderr)
        time.sleep(delay)
