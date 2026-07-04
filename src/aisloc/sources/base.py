"""Provider abstraction.

The gathering pipeline only ever sees ``RepoRef`` objects and a ``RepoSource``
iterator. Switching from public GitHub to an on-prem GitLab means writing one
new ``RepoSource`` subclass; nothing in ``mining`` or ``gather`` changes.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RepoRef:
    """Everything the miner needs to fetch and identify one repository."""

    provider: str
    repo_id: str  # stable id within the provider, used for dedup/resume
    name: str  # human-readable "owner/name"
    clone_url: str  # https or ssh URL git can clone
    default_branch: str | None = None
    # Provider metadata carried through to the record for later covariates.
    meta: dict[str, object] = field(default_factory=dict)


class RepoSource(abc.ABC):
    """A stream of repositories to mine, plus auth for cloning.

    Implementations must be *lazy* (yield as they page) so the orchestrator can
    start work immediately and stop early once ``target_repos`` is reached.
    """

    #: short provider tag stored in every record
    name: str = "base"

    @abc.abstractmethod
    def iter_repos(self) -> Iterator[RepoRef]:
        """Yield candidate repos, best-effort ordered by relevance/activity."""

    def clone_env(self) -> dict[str, str]:
        """Extra environment for the clone subprocess (e.g. credential helper,
        askpass, tokens). Default: none (anonymous https)."""
        return {}

    def authorize_url(self, ref: RepoRef) -> str:
        """Return the URL to actually clone. Hook for injecting tokens into the
        URL for private forges; default returns the public URL unchanged."""
        return ref.clone_url
