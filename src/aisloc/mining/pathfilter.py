"""Exclude non-human / generated / vendored paths from churn.

The single biggest SLOC-inflation trap (concept.md sec. 5.3): if lockfiles,
vendored trees, generated code and minified assets count as "AI velocity", the
whole result is noise. We drop them here, before any aggregation.
"""

from __future__ import annotations

import re

# Directory segments that mark vendored / third-party / dependency trees.
_VENDOR_DIRS = {
    "vendor",
    "third_party",
    "third-party",
    "node_modules",
    "bower_components",
    ".venv",
    "venv",
    "dist",
    "build",
    "out",
    "target",
    ".git",
    "Pods",
    "Carthage",
    "external",
    "deps",
    "godeps",
    "site-packages",
}

# Exact basenames that are machine-managed (lockfiles, manifests).
_LOCK_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    "Gemfile.lock",
    "go.sum",
    "flake.lock",
    "packages.lock.json",
    "gradle.lockfile",
}

# Suffix patterns for generated / minified / binary-ish content.
_GENERATED_SUFFIXES = (
    ".min.js",
    ".min.css",
    ".map",
    ".pb.go",
    "_pb2.py",
    "_pb2_grpc.py",
    ".pb.cc",
    ".pb.h",
    ".g.dart",
    ".freezed.dart",
    ".designer.cs",
    ".generated.go",
)

# Path fragments that usually mark generated or non-source material.
_GENERATED_RE = re.compile(
    r"(^|/)(dist|build|generated|__generated__|migrations|snapshots|__snapshots__"
    r"|fixtures|testdata|vendored|gen)(/|$)",
    re.IGNORECASE,
)


def is_excluded(path: str) -> bool:
    """True if this path must not contribute to source churn."""
    if not path:
        return True
    segments = path.split("/")
    if any(seg in _VENDOR_DIRS for seg in segments):
        return True
    base = segments[-1]
    if base in _LOCK_FILES:
        return True
    if base.endswith(_GENERATED_SUFFIXES):
        return True
    if _GENERATED_RE.search(path):
        return True
    return False
