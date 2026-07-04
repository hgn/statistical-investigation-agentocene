"""Mine one repository into a self-contained record.

Sequence per repo: bounded clone -> churn pass -> signature pass -> delete
clone. Author emails are pseudonymised (salted hash) at serialisation so the
same developer is joinable across the churn and signature views without storing
raw addresses, which also eases the move to the on-prem/company deployment.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..sources import RepoRef, RepoSource
from . import churn, gitio, signatures

_SALT = os.environ.get("AISLOC_SALT", "aisloc-v1")


def hash_email(email: str) -> str:
    h = hashlib.sha256(f"{_SALT}\x00{email.lower()}".encode()).hexdigest()
    return h[:16]


async def mine_repo(ref: RepoRef, cfg: Config, source: RepoSource, dest: Path) -> dict[str, object]:
    url = source.authorize_url(ref)
    env = source.clone_env()

    clone = await gitio.clone_bounded(url, dest, cfg, env)
    try:
        agg = churn.Aggregator()
        async for line in gitio.stream_log(dest, churn.log_args(cfg.baseline_since), cfg, env):
            churn.feed_line(agg, line)
        churn_rec = agg.to_record()

        sig_lines = gitio.stream_log(dest, signatures.log_args(cfg.baseline_since), cfg, env)
        sig_rec = await signatures.scan(sig_lines)
    finally:
        gitio.cleanup(dest)

    return _assemble(ref, cfg, clone.shallow, churn_rec, sig_rec)


def _assemble(
    ref: RepoRef,
    cfg: Config,
    shallow: bool,
    churn_rec: dict[str, object],
    sig_rec: dict[str, object],
) -> dict[str, object]:
    # Pseudonymise emails consistently across both views.
    authors = []
    for row in churn_rec["authors"]:  # type: ignore[union-attr]
        r = dict(row)  # type: ignore[arg-type]
        r["dev"] = hash_email(str(r.pop("email")))
        authors.append(r)

    sig_authors = {
        hash_email(email): classes
        for email, classes in (sig_rec.get("authors") or {}).items()  # type: ignore[union-attr]
    }

    return {
        "schema": 1,
        "provider": ref.provider,
        "repo_id": ref.repo_id,
        "name": ref.name,
        "clone_url": ref.clone_url,
        "default_branch": ref.default_branch,
        "gathered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_since": cfg.baseline_since,
        "shallow": shallow,
        "meta": ref.meta,
        "signatures": {"repo": sig_rec.get("repo", {}), "authors": sig_authors},
        "months": churn_rec["months"],
        "activity": churn_rec["activity"],
        "authors": authors,
        "status": "ok",
    }
