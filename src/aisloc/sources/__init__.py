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
            kwargs: dict[str, object] = {}
            if "query" in opts:
                kwargs["query"] = opts["query"]
            if "max_pages" in opts:
                kwargs["max_pages"] = int(opts["max_pages"])
            if "require_code_language" in opts:
                kwargs["require_code_language"] = opts["require_code_language"] not in (
                    "0", "false", "False",
                )
            if "min_recent_commits" in opts:
                kwargs["min_recent_commits"] = int(opts["min_recent_commits"])
            if "recent_window_days" in opts:
                kwargs["recent_window_days"] = int(opts["recent_window_days"])
            return GitHubSource(token=opts.get("token"), **kwargs)  # type: ignore[arg-type]
        case "gitlab":
            gl_kwargs: dict[str, object] = {}
            if "max_pages" in opts:
                gl_kwargs["max_pages"] = int(opts["max_pages"])
            if "include_archived" in opts:
                gl_kwargs["include_archived"] = opts["include_archived"] not in (
                    "0", "false", "False",
                )
            if "verify_ssl" in opts:
                gl_kwargs["verify_ssl"] = opts["verify_ssl"] not in ("0", "false", "False")
            return GitLabSource(
                base_url=opts["base_url"],
                token=opts.get("token"),
                group=opts.get("group"),
                **gl_kwargs,  # type: ignore[arg-type]
            )
        case "list":
            return ListSource(path=opts["path"], provider_tag=opts.get("tag", "github"))
        case other:
            raise ValueError(f"unknown provider: {other!r}")
