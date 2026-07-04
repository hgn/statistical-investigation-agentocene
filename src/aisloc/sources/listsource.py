"""File-backed repository source.

Reads a newline- or JSON-delimited list of repos produced offline (e.g. a
stratified GH Archive / BigQuery sample, per concept.md sec. 1). This is the
source the real study uses, because it lets sampling be a separate, auditable,
reproducible step rather than a live API ranking.

Accepted line formats:
  * a bare clone URL:            https://github.com/owner/name.git
  * ``owner/name``               (assumed GitHub https)
  * a JSON object per line:      {"clone_url": "...", "name": "...", "meta": {...}}
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .base import RepoRef, RepoSource


class ListSource(RepoSource):
    name = "list"

    def __init__(self, path: str, provider_tag: str = "github") -> None:
        self._path = Path(path)
        self._tag = provider_tag

    def iter_repos(self) -> Iterator[RepoRef]:
        with self._path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                yield self._parse(line, lineno)

    def _parse(self, line: str, lineno: int) -> RepoRef:
        if line.startswith("{"):
            obj = json.loads(line)
            url = obj["clone_url"]
            name = obj.get("name") or _name_from_url(url)
            return RepoRef(
                provider=self._tag,
                repo_id=str(obj.get("repo_id") or name),
                name=name,
                clone_url=url,
                default_branch=obj.get("default_branch"),
                meta=dict(obj.get("meta") or {}),
            )
        if "://" in line or line.startswith("git@"):
            url, name = line, _name_from_url(line)
        else:  # owner/name
            url, name = f"https://github.com/{line}.git", line
        return RepoRef(provider=self._tag, repo_id=name, name=name, clone_url=url)


def _name_from_url(url: str) -> str:
    tail = url.rstrip("/").split("/")[-2:]
    return "/".join(tail).removesuffix(".git")
