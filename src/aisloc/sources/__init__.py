"""Repository sources and the factory that selects one from config."""

from __future__ import annotations

from ..config import Config
from .base import RepoRef, RepoSource
from .github import GitHubSource
from .gitlab import GitLabSource
from .listsource import ListSource

__all__ = ["RepoRef", "RepoSource", "make_source"]


def make_source(cfg: Config) -> RepoSource:
    """Instantiate the configured provider. Adding a new forge = one branch."""
    opts = cfg.provider_opts
    match cfg.provider:
        case "github":
            return GitHubSource(
                query=opts.get("query", "stars:>100 pushed:>2024-01-01"),
                token=opts.get("token"),
                max_pages=int(opts.get("max_pages", 10)),
            )
        case "gitlab":
            return GitLabSource(
                base_url=opts["base_url"],
                token=opts.get("token"),
                group=opts.get("group"),
            )
        case "list":
            return ListSource(path=opts["path"], provider_tag=opts.get("tag", "github"))
        case other:
            raise ValueError(f"unknown provider: {other!r}")
