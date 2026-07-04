"""Git mining: bounded clone, churn/signature extraction, per-repo records."""

from __future__ import annotations

from .miner import mine_repo

__all__ = ["mine_repo"]
