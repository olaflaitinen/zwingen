#!/usr/bin/env python3
# tools/check_dashes.py
# Maintainer: @olaflaitinen <olaf.laitinen@gmail.com>
# License: Apache-2.0 (matches the host project)
#
# Reject typographic dashes in public-facing text. The zwingen style guide
# accepts only the plain ASCII hyphen-minus (U+002D, '-'). All other
# dash-like characters are rewritten to '-' (with `--fix`) or reported as
# violations (default).
#
# Forbidden codepoints:
#   U+2010 hyphen
#   U+2011 non-breaking hyphen
#   U+2012 figure dash
#   U+2013 en dash
#   U+2014 em dash
#   U+2015 horizontal bar
#   U+2212 minus sign
#   U+FE58 small em dash
#   U+FE63 small hyphen-minus
#   U+FF0D fullwidth hyphen-minus
#
# Scope is the same as `tools/lint_no_emoji.py`: README, CHANGELOG, docs/,
# notebook markdown cells, and module/class/function docstrings under
# src/zwingen/. Code lines (non-docstring) are skipped so we do not break
# legitimate uses such as `x = a - b`.
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Final

FORBIDDEN: Final[dict[int, str]] = {
    0x2010: "HYPHEN",
    0x2011: "NON-BREAKING HYPHEN",
    0x2012: "FIGURE DASH",
    0x2013: "EN DASH",
    0x2014: "EM DASH",
    0x2015: "HORIZONTAL BAR",
    0x2212: "MINUS SIGN",
    0xFE58: "SMALL EM DASH",
    0xFE63: "SMALL HYPHEN-MINUS",
    0xFF0D: "FULLWIDTH HYPHEN-MINUS",
}
_FORBIDDEN_CHARS: Final[frozenset[str]] = frozenset(chr(cp) for cp in FORBIDDEN)
_REPLACEMENT: Final[str] = "-"


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
            f"forbidden dash {self.codepoint} ({self.name}) in: {self.snippet!r}"
        )


def _scan_text(path: pathlib.Path, text: str, line_offset: int = 0) -> Iterator[Finding]:
    for line_no, line in enumerate(text.splitlines(), start=1):
        for col_no, char in enumerate(line, start=1):
            if char in _FORBIDDEN_CHARS:
                yield Finding(
                    path=str(path),
                    line=line_offset + line_no,
                    column=col_no,
                    codepoint=f"U+{ord(char):04X}",
                    name=FORBIDDEN[ord(char)],
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


def _iter_notebook_markdown(path: pathlib.Path) -> Iterator[tuple[int, int, str]]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    cells = nb.get("cells", [])
    for cell_idx, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        yield cell_idx, 0, src


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


def _replace_in_text(text: str) -> tuple[str, int]:
    """Return (rewritten, count). Replaces every forbidden codepoint with '-'."""
    count = 0
    out: list[str] = []
    for ch in text:
        if ch in _FORBIDDEN_CHARS:
            out.append(_REPLACEMENT)
            count += 1
        else:
            out.append(ch)
    return "".join(out), count


def lint(root: pathlib.Path, *, fix: bool) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    fixed_total = 0
    for path in collect_files(root):
        rel = path.relative_to(root)
        if path.suffix == ".py":
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for start, doc in _iter_docstrings(path, source):
                findings.extend(_scan_text(rel, doc, line_offset=start - 1))
            # We do NOT auto-fix .py source: legitimate `-` operators must
            # not be touched. Maintainers fix docstrings by hand.
        elif path.suffix == ".ipynb":
            try:
                nb = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            mutated = False
            for cell_idx, _, src in _iter_notebook_markdown(path):
                tagged_path = pathlib.Path(f"{rel}#cell={cell_idx}")
                findings.extend(_scan_text(tagged_path, src))
                if fix:
                    rewritten, count = _replace_in_text(src)
                    if count:
                        nb["cells"][cell_idx - 1]["source"] = rewritten
                        fixed_total += count
                        mutated = True
            if fix and mutated:
                path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            findings.extend(_scan_text(rel, text))
            if fix:
                rewritten, count = _replace_in_text(text)
                if count:
                    path.write_text(rewritten, encoding="utf-8")
                    fixed_total += count
    return findings, fixed_total


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.check_dashes",
        description="Reject typographic dashes in public-facing zwingen prose.",
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Repository root (defaults to CWD).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rewrite forbidden dashes to plain '-'. Skips .py source files.",
    )
    parser.add_argument(
        "--report",
        choices=("jsonl", "human"),
        default="human",
        help="Output format.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    findings, fixed = lint(args.root.resolve(), fix=args.fix)
    out = sys.stdout
    if args.report == "jsonl":
        for f in findings:
            out.write(f.to_json())
            out.write("\n")
    else:
        for f in findings:
            out.write(f.to_human())
            out.write("\n")
        if args.fix and fixed:
            out.write(f"\nRewrote {fixed} forbidden dash(es) to '-'.\n")
        if findings and not args.fix:
            out.write(f"\n{len(findings)} forbidden-dash finding(s).\n")
        if not findings:
            out.write("No forbidden dashes found.\n")
    if args.fix:
        # When fixing, only .py docstring findings remain unaddressed.
        unfixed = [f for f in findings if f.path.endswith(".py")]
        return 1 if unfixed else 0
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
