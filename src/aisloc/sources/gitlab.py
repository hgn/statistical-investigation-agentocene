"""On-prem GitLab repository source (stub for the company deployment).

The whole point of the ``RepoSource`` abstraction: to switch the study from
public GitHub to the internal GitLab, only this file needs finishing. The miner,
governor, orchestrator and output schema are unchanged.

To complete it:
* implement ``iter_repos`` against the GitLab REST API
  (``GET {base_url}/api/v4/projects?membership=true&per_page=100``, paginated
  via the ``X-Next-Page`` header), mapping each project to a ``RepoRef``;
* embed the token for cloning via ``authorize_url`` (GitLab supports
  ``https://oauth2:<token>@host/group/repo.git``) or set a credential helper in
  ``clone_env``.

Everything else already works.
"""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Iterator

from .base import RepoRef, RepoSource


class GitLabSource(RepoSource):
    name = "gitlab"

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        group: str | None = None,
        per_page: int = 100,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token or os.environ.get("GITLAB_TOKEN")
        self._group = group
        self._per_page = min(per_page, 100)

    def iter_repos(self) -> Iterator[RepoRef]:  # pragma: no cover - stub
        raise NotImplementedError(
            "GitLabSource.iter_repos is a stub for the on-prem deployment; "
            "implement against GET /api/v4/projects (see module docstring)."
        )

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
