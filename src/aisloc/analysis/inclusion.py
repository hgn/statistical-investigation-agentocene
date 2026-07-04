"""Pre-registered inclusion gate (concept.md sec. 5.1).

A repo enters the panel only if it can actually inform a pre/post-AI contrast.
The gate is deterministic and returns the list of failed criteria, so the
manifest can explain every exclusion (no silent dropping = no p-hacking).
"""

from __future__ import annotations

from dataclasses import dataclass

from .records import RepoSummary, ym_ord

# Global AI availability anchor. The study identifies off this calendar shock
# (concept.md sec. 2.2), so inclusion requires spanning it with margin.
DEFAULT_ANCHOR = "2022-11"  # ChatGPT inflection


@dataclass(frozen=True)
class InclusionRule:
    anchor: str = DEFAULT_ANCHOR
    min_pre_months: int = 18
    min_post_months: int = 9
    min_active_ratio: float = 0.70
    max_gap_months: int = 3
    min_contributors: int = 3
    min_commits: int = 200


@dataclass(frozen=True)
class Verdict:
    included: bool
    reasons: list[str]  # failed criteria (empty if included)


def evaluate(s: RepoSummary, rule: InclusionRule = InclusionRule()) -> Verdict:
    reasons: list[str] = []
    anchor = ym_ord(rule.anchor)

    if not s.first_month or not s.last_month:
        return Verdict(False, ["no-activity"])

    pre = anchor - ym_ord(s.first_month)
    post = ym_ord(s.last_month) - anchor
    if pre < rule.min_pre_months:
        reasons.append(f"pre-span<{rule.min_pre_months}m (has {pre})")
    if post < rule.min_post_months:
        reasons.append(f"post-span<{rule.min_post_months}m (has {post})")

    active_ratio = s.active_months / s.span_months if s.span_months else 0.0
    if active_ratio < rule.min_active_ratio:
        reasons.append(f"active-ratio<{rule.min_active_ratio:.0%} (has {active_ratio:.0%})")
    if s.max_gap_months > rule.max_gap_months:
        reasons.append(f"gap>{rule.max_gap_months}m (has {s.max_gap_months})")
    if s.contributors < rule.min_contributors:
        reasons.append(f"contributors<{rule.min_contributors} (has {s.contributors})")
    if s.commits < rule.min_commits:
        reasons.append(f"commits<{rule.min_commits} (has {s.commits})")

    return Verdict(not reasons, reasons)
