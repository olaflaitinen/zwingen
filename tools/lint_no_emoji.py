#!/usr/bin/env python3
# tools/lint_no_emoji.py
# Maintainer: @olaflaitinen <olaf.laitinen@gmail.com>
# License: Apache-2.0 (matches the host project)
#
# Reject emoji codepoints in public-facing text. The zwingen style guide
# forbids emoji in:
#   - README.md, CHANGELOG.md, CITATION.cff, GOVERNANCE.md, MAINTAINERS.md,
#     SECURITY.md, CONTRIBUTING.md, ROADMAP.md, NOTICE
#   - everything under docs/
#   - the markdown cells of every notebook under examples/
#   - module, class, and function docstrings under src/zwingen/
#
# Code identifiers and comments inside .py source are scanned by ruff's RUF001
# / RUF002 family; this script intentionally focuses on prose surfaces so that
# we do not double-flag the same finding.
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Final

# Unicode ranges that count as "emoji" for our purposes. We deliberately do
# not rely on the `emoji` PyPI package: it pulls in 1.6 MB of data tables and
# we only need the prose-relevant subset.
_EMOJI_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x1F1E6, 0x1F1FF),  # regional indicators (flags)
    (0x1F300, 0x1F5FF),  # misc symbols and pictographs
    (0x1F600, 0x1F64F),  # emoticons
    (0x1F680, 0x1F6FF),  # transport and map
    (0x1F700, 0x1F77F),  # alchemical
    (0x1F780, 0x1F7FF),  # geometric shapes extended
    (0x1F800, 0x1F8FF),  # supplemental arrows-c
    (0x1F900, 0x1F9FF),  # supplemental symbols and pictographs
    (0x1FA00, 0x1FA6F),  # chess, symbols and pictographs extended-a
    (0x1FA70, 0x1FAFF),  # symbols and pictographs extended-b
    (0x2600, 0x26FF),    # misc symbols
    (0x2700, 0x27BF),    # dingbats
    (0x2300, 0x23FF),    # misc technical (clocks, hourglasses, etc.)
)

# Codepoints that, while inside the ranges above, we explicitly allow because
# they are routinely used in academic prose (mathematical, currency, arrows).
_ALLOWLIST: Final[frozenset[int]] = frozenset({
    0x2192,  # right arrow used in pseudo-code
    0x2190,  # left arrow
    0x2194,  # left-right arrow
    0x21D2,  # double right arrow (implies)
    0x2208,  # element of
    0x2209,  # not element of
    0x222B,  # integral
    0x2211,  # n-ary summation
    0x220F,  # n-ary product
    0x2202,  # partial differential
    0x2207,  # nabla
    0x221E,  # infinity
    0x2248,  # almost equal
    0x2260,  # not equal
    0x2264,  # less than or equal
    0x2265,  # greater than or equal
})

_VARIATION_SELECTOR_16: Final[int] = 0xFE0F
_ZERO_WIDTH_JOINER: Final[int] = 0x200D


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    column: int
    codepoint: str
    name: str
    snippet: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def to_human(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.column}: "
            f"emoji {self.codepoint} ({self.name}) in: {self.snippet!r}"
        )


def _is_emoji(cp: int) -> bool:
    if cp in _ALLOWLIST:
        return False
    if cp in (_VARIATION_SELECTOR_16, _ZERO_WIDTH_JOINER):
        return False
    return any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES)


def _scan_text(path: pathlib.Path, text: str) -> Iterator[Finding]:
    for line_no, line in enumerate(text.splitlines(), start=1):
        for col_no, char in enumerate(line, start=1):
            cp = ord(char)
            if _is_emoji(cp):
                try:
                    name = unicodedata.name(char)
                except ValueError:
                    name = "UNNAMED"
                yield Finding(
                    path=str(path),
                    line=line_no,
                    column=col_no,
                    codepoint=f"U+{cp:04X}",
                    name=name,
                    snippet=line.strip()[:120],
                )


def _iter_docstrings(path: pathlib.Path, source: str) -> Iterator[tuple[int, str]]:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                start = getattr(node, "lineno", 1)
                yield start, doc


def _iter_notebook_markdown(path: pathlib.Path) -> Iterator[tuple[int, str]]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for cell_idx, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        yield cell_idx, src


_PROSE_GLOBS: Final[tuple[str, ...]] = (
    "README.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "MAINTAINERS.md",
    "NOTICE",
    "ROADMAP.md",
    "SECURITY.md",
    "docs/**/*.md",
    "docs/**/*.rst",
)
_NOTEBOOK_GLOB: Final[str] = "examples/**/*.ipynb"
_PYTHON_GLOB: Final[str] = "src/zwingen/**/*.py"


def collect_files(root: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for pattern in _PROSE_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    files.extend(sorted(root.glob(_NOTEBOOK_GLOB)))
    files.extend(sorted(root.glob(_PYTHON_GLOB)))
    return [f for f in files if f.is_file()]


def lint(root: pathlib.Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in collect_files(root):
        rel = path.relative_to(root)
        if path.suffix == ".py":
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for start, doc in _iter_docstrings(path, source):
                for f in _scan_text(rel, doc):
                    findings.append(
                        Finding(
                            path=f.path,
                            line=start + f.line - 1,
                            column=f.column,
                            codepoint=f.codepoint,
                            name=f.name,
                            snippet=f.snippet,
                        )
                    )
        elif path.suffix == ".ipynb":
            for cell_idx, src in _iter_notebook_markdown(path):
                tagged_path = pathlib.Path(f"{rel}#cell={cell_idx}")
                findings.extend(_scan_text(tagged_path, src))
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            findings.extend(_scan_text(rel, text))
    return findings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.lint_no_emoji",
        description="Reject emoji in public-facing zwingen prose.",
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Repository root (defaults to CWD).",
    )
    parser.add_argument(
        "--report",
        choices=("jsonl", "human"),
        default="human",
        help="Output format.",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=0,
        help="Stop after N findings (0 = unlimited).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    findings = lint(args.root.resolve())
    if args.max_findings > 0:
        findings = findings[: args.max_findings]
    out = sys.stdout
    if args.report == "jsonl":
        for f in findings:
            out.write(f.to_json())
            out.write("\n")
    else:
        for f in findings:
            out.write(f.to_human())
            out.write("\n")
        if findings:
            out.write(f"\n{len(findings)} emoji finding(s).\n")
        else:
            out.write("No emoji found in public-facing prose.\n")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
