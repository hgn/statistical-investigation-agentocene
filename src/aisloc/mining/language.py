"""Map file paths to a language and an AI-suitability score.

The AI-suitability score is the *identifying* variable of the whole study
(concept.md sec. 2.3): a genuine AI effect must be larger where coding assistants
are strong (mainstream languages with abundant training data) and near-zero for
niche languages they barely support. The scores below are an explicit,
pre-registered modelling assumption on a 0..1 scale, not ground truth; the
dose-response test is robust to the exact values as long as the ordering holds.
"""

from __future__ import annotations

# extension (without dot, lowercase) -> canonical language name
_EXT_LANG: dict[str, str] = {
    "py": "python", "pyi": "python",
    "js": "javascript", "jsx": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "java": "java",
    "go": "go",
    "rb": "ruby",
    "php": "php",
    "cs": "csharp",
    "c": "c", "h": "c",
    "cc": "cpp", "cpp": "cpp", "cxx": "cpp", "hpp": "cpp", "hh": "cpp",
    "rs": "rust",
    "kt": "kotlin", "kts": "kotlin",
    "swift": "swift",
    "scala": "scala", "sc": "scala",
    "dart": "dart",
    "sql": "sql",
    "sh": "shell", "bash": "shell", "zsh": "shell",
    "html": "html", "htm": "html",
    "css": "css", "scss": "css", "sass": "css",
    "yaml": "yaml", "yml": "yaml",
    "r": "r",
    "m": "objc", "mm": "objc",
    "lua": "lua",
    "pl": "perl", "pm": "perl",
    "ex": "elixir", "exs": "elixir",
    "erl": "erlang",
    "hs": "haskell",
    "ml": "ocaml", "mli": "ocaml",
    "clj": "clojure",
    "nix": "nix",
    "tcl": "tcl",
    "vhd": "vhdl", "vhdl": "vhdl",
    "v": "verilog", "sv": "verilog",
    "adb": "ada", "ads": "ada",
    "f": "fortran", "f90": "fortran", "f95": "fortran", "for": "fortran",
    "cob": "cobol", "cbl": "cobol",
    "asm": "assembly", "s": "assembly",
}

# language -> AI-suitability in [0,1]. High = mainstream, huge training corpus,
# first-class assistant support. Low = niche/legacy, weak assistant support.
_SUITABILITY: dict[str, float] = {
    "python": 1.00, "javascript": 1.00, "typescript": 1.00,
    "java": 0.90, "go": 0.90, "csharp": 0.90, "php": 0.85, "ruby": 0.85,
    "cpp": 0.80, "c": 0.75, "rust": 0.80, "kotlin": 0.80, "swift": 0.78,
    "scala": 0.70, "dart": 0.70,
    "sql": 0.65, "shell": 0.60, "html": 0.60, "css": 0.55, "r": 0.60,
    "yaml": 0.45, "lua": 0.50, "objc": 0.55, "perl": 0.45,
    "elixir": 0.50, "haskell": 0.45, "clojure": 0.45, "ocaml": 0.40,
    "erlang": 0.35,
    "nix": 0.30, "tcl": 0.25, "assembly": 0.25,
    "vhdl": 0.15, "verilog": 0.15, "ada": 0.15, "fortran": 0.20, "cobol": 0.10,
}

OTHER = "other"
OTHER_SUITABILITY = 0.40  # unknown extensions: mid, treated cautiously


def classify(path: str) -> str:
    """Canonical language for a path, or ``OTHER``."""
    base = path.rsplit("/", 1)[-1]
    if "." not in base:
        return OTHER
    ext = base.rsplit(".", 1)[-1].lower()
    return _EXT_LANG.get(ext, OTHER)


def suitability(language: str) -> float:
    return _SUITABILITY.get(language, OTHER_SUITABILITY)


def is_source(language: str) -> bool:
    """Whether the language counts as source code (excludes pure markup/config
    from the primary churn outcome; they remain available as negative controls)."""
    return language not in {OTHER, "yaml", "html", "css"}
