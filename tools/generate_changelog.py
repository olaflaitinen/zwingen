#!/usr/bin/env python3
# tools/generate_changelog.py
# Maintainer: @olaflaitinen <olaf.laitinen@gmail.com>
# License: Apache-2.0 (matches the host project)
#
# Build a Keep-a-Changelog (https://keepachangelog.com) document from the
# Conventional Commits log between two refs. The output groups commits by
# semantic section, attributes them to GitHub PRs when discoverable, and
# emits two artefacts:
#   1. a Markdown block to be prepended to CHANGELOG.md (or replace it);
#   2. a release-notes JSON file consumed by .github/workflows/release.yml.
#
# Sections (in order):
#   Added       feat:
#   Changed     refactor:, perf:
#   Deprecated  deprecate:
#   Removed     remove:
#   Fixed       fix:
#   Security    security:
#   Numerical   numerical: (zwingen-specific tolerance changes)
#   Docs        docs:
#   Build       build:, ci:, chore:
#
# Breaking changes (BREAKING CHANGE: footer or `!` after type) are surfaced
# in a top-level "BREAKING CHANGES" callout.
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

_TYPE_TO_SECTION: Final[OrderedDict[str, str]] = OrderedDict(
    [
        ("feat", "Added"),
        ("refactor", "Changed"),
        ("perf", "Changed"),
        ("deprecate", "Deprecated"),
        ("remove", "Removed"),
        ("fix", "Fixed"),
        ("security", "Security"),
        ("numerical", "Numerical"),
        ("docs", "Docs"),
        ("build", "Build"),
        ("ci", "Build"),
        ("chore", "Build"),
    ]
)
_SECTION_ORDER: Final[tuple[str, ...]] = (
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Numerical",
    "Docs",
    "Build",
)

_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<bang>!)?:\s+(?P<subject>.+)$"
)
_PR_RE: Final[re.Pattern[str]] = re.compile(r"\(#(\d+)\)\s*$")
_BREAKING_RE: Final[re.Pattern[str]] = re.compile(
    r"^BREAKING CHANGE:\s*(?P<body>.+)$", re.MULTILINE
)


@dataclass(frozen=True, slots=True)
class Commit:
    sha: str
    type: str
    scope: str | None
    subject: str
    breaking: str | None
    pr: int | None
    raw_body: str


@dataclass(slots=True)
class Section:
    title: str
    entries: list[str] = field(default_factory=list)


def _git(*args: str, cwd: pathlib.Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _commit_log(
    repo: pathlib.Path, range_spec: str
) -> Iterable[tuple[str, str]]:
    sep = "\x1e"  # ASCII record separator
    fmt = f"%H{sep}%B%x00"
    raw = _git("log", "--format=" + fmt, range_spec, cwd=repo)
    for record in raw.split("\x00"):
        record = record.strip("\n")
        if not record:
            continue
        sha, _, body = record.partition(sep)
        yield sha, body


def _parse_commit(sha: str, body: str) -> Commit | None:
    lines = body.splitlines()
    if not lines:
        return None
    header = lines[0]
    m = _HEADER_RE.match(header)
    if not m:
        return None
    pr_match = _PR_RE.search(m["subject"])
    pr = int(pr_match[1]) if pr_match else None
    subject = _PR_RE.sub("", m["subject"]).strip()
    breaking: str | None = None
    if m["bang"]:
        breaking = subject
    breaking_footer = _BREAKING_RE.search(body)
    if breaking_footer:
        breaking = breaking_footer["body"].strip()
    return Commit(
        sha=sha,
        type=m["type"],
        scope=m["scope"],
        subject=subject,
        breaking=breaking,
        pr=pr,
        raw_body=body,
    )


def _format_entry(c: Commit) -> str:
    scope = f"**{c.scope}**: " if c.scope else ""
    pr_link = f" ([#{c.pr}](https://github.com/olaflaitinen/zwingen/pull/{c.pr}))" if c.pr else ""
    return f"- {scope}{c.subject}{pr_link}"


def build_changelog(
    commits: Iterable[Commit],
    *,
    version: str,
    release_date: dt.date,
    repo_url: str,
    previous_tag: str | None,
) -> tuple[str, dict[str, object]]:
    sections: dict[str, Section] = {
        title: Section(title=title) for title in _SECTION_ORDER
    }
    breaking: list[str] = []
    for c in commits:
        section_title = _TYPE_TO_SECTION.get(c.type)
        if section_title is None:
            continue
        sections[section_title].entries.append(_format_entry(c))
        if c.breaking:
            breaking.append(
                f"- **{c.scope or c.type}**: {c.breaking} (commit `{c.sha[:7]}`)"
            )
    lines: list[str] = []
    compare_url = (
        f"{repo_url}/compare/{previous_tag}...v{version}"
        if previous_tag
        else f"{repo_url}/releases/tag/v{version}"
    )
    lines.append(f"## [{version}]({compare_url}) - {release_date.isoformat()}")
    lines.append("")
    if breaking:
        lines.append("### BREAKING CHANGES")
        lines.extend(breaking)
        lines.append("")
    for title in _SECTION_ORDER:
        section = sections[title]
        if not section.entries:
            continue
        lines.append(f"### {title}")
        lines.extend(section.entries)
        lines.append("")
    body = "\n".join(lines).rstrip() + "\n"
    payload: dict[str, object] = {
        "version": version,
        "release_date": release_date.isoformat(),
        "compare_url": compare_url,
        "breaking": breaking,
        "sections": {
            title: section.entries for title, section in sections.items() if section.entries
        },
    }
    return body, payload


def _resolve_previous_tag(repo: pathlib.Path, current_tag: str) -> str | None:
    try:
        out = _git("describe", "--tags", "--abbrev=0", f"{current_tag}^", cwd=repo)
        return out or None
    except subprocess.CalledProcessError:
        return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.generate_changelog",
        description="Generate Keep-a-Changelog entries from Conventional Commits.",
    )
    parser.add_argument("--version", required=True, help="Target release version, e.g. 0.1.0.")
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument(
        "--repo-url",
        default="https://github.com/olaflaitinen/zwingen",
        help="Canonical repository URL (used in compare links).",
    )
    parser.add_argument(
        "--from-ref",
        default=None,
        help="Lower bound of the commit range. Defaults to the previous tag.",
    )
    parser.add_argument(
        "--to-ref",
        default="HEAD",
        help="Upper bound of the commit range. Defaults to HEAD.",
    )
    parser.add_argument(
        "--changelog",
        type=pathlib.Path,
        default=pathlib.Path("CHANGELOG.md"),
        help="Path to the changelog file to update.",
    )
    parser.add_argument(
        "--release-notes-json",
        type=pathlib.Path,
        default=pathlib.Path("release-notes.json"),
        help="Where to write the structured release-notes payload.",
    )
    parser.add_argument(
        "--release-date",
        default=None,
        help="ISO date for the release (default: today, UTC).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print to stdout instead of mutating files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = args.repo.resolve()
    current_tag = f"v{args.version}"
    previous_tag = args.from_ref or _resolve_previous_tag(repo, current_tag)
    range_spec = f"{previous_tag}..{args.to_ref}" if previous_tag else args.to_ref
    commits: list[Commit] = []
    for sha, body in _commit_log(repo, range_spec):
        parsed = _parse_commit(sha, body)
        if parsed is not None:
            commits.append(parsed)
    release_date = (
        dt.date.fromisoformat(args.release_date)
        if args.release_date
        else dt.datetime.now(tz=dt.UTC).date()
    )
    body, payload = build_changelog(
        commits,
        version=args.version,
        release_date=release_date,
        repo_url=args.repo_url,
        previous_tag=previous_tag,
    )
    if args.dry_run:
        sys.stdout.write(body)
        sys.stdout.write("\n--- payload ---\n")
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False))
        sys.stdout.write("\n")
        return 0
    args.release_notes_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    changelog_path = args.changelog
    existing = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    header = "# Changelog\n\nAll notable changes follow [Keep a Changelog](https://keepachangelog.com).\n\n"
    rest = existing
    if existing.startswith("# Changelog"):
        # Strip the existing header so we can re-emit it once.
        rest = existing.split("\n", 4)[-1]
    changelog_path.write_text(header + body + "\n" + rest, encoding="utf-8")
    sys.stdout.write(f"Wrote {changelog_path} and {args.release_notes_json}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
