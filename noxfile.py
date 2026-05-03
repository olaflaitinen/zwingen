# noxfile.py
# Maintainer: @olaflaitinen <olaf.laitinen@gmail.com>
# License: Apache-2.0
from __future__ import annotations

import os
import pathlib
import shutil
import sys
from typing import Final

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = True
nox.options.sessions = (
    "lint",
    "typecheck",
    "style_guard",
    "tests",
    "docs",
)

PACKAGE: Final[str] = "zwingen"
ROOT: Final[pathlib.Path] = pathlib.Path(__file__).parent
SRC: Final[pathlib.Path] = ROOT / "src" / PACKAGE
DOCS: Final[pathlib.Path] = ROOT / "docs"
TOOLS: Final[pathlib.Path] = ROOT / "tools"

STABLE_PYTHONS: Final[tuple[str, ...]] = ("3.10", "3.11", "3.12", "3.13")
EXPERIMENTAL_PYTHONS: Final[tuple[str, ...]] = ("3.14", "3.15", "3.15t")
ALL_PYTHONS: Final[tuple[str, ...]] = STABLE_PYTHONS + EXPERIMENTAL_PYTHONS


def _install(session: nox.Session, *extras: str) -> None:
    spec = f".[{','.join(extras)}]" if extras else "."
    session.install("-e", spec)


@nox.session(python=STABLE_PYTHONS)
def tests(session: nox.Session) -> None:
    """Run the standard test matrix."""
    _install(session, "test")
    session.run(
        "pytest",
        "-m",
        "not slow and not gpu",
        "-n",
        "auto",
        "--cov=zwingen",
        "--cov-report=xml",
        *session.posargs,
    )


@nox.session(python=EXPERIMENTAL_PYTHONS, tags=["experimental"])
def tests_experimental(session: nox.Session) -> None:
    """Run tests on free-threaded and bleeding-edge Python builds."""
    _install(session, "test")
    env = {
        "PYTHON_GIL": "0" if session.python.endswith("t") else "1",
        "ZWINGEN_EXPERIMENTAL": "1",
    }
    session.run(
        "pytest",
        "-m",
        "not gpu",
        "--no-cov",
        *session.posargs,
        env=env,
    )


@nox.session(python="3.12")
def lint(session: nox.Session) -> None:
    """Run ruff (check + format check)."""
    session.install("ruff>=0.5")
    session.run("ruff", "check", "src", "tests", "tools")
    session.run("ruff", "format", "--check", "src", "tests", "tools")


@nox.session(python="3.12")
def typecheck(session: nox.Session) -> None:
    """Run mypy strict and pyright strict."""
    _install(session, "dev")
    session.run("mypy", "src/zwingen", "tools")
    session.run("pyright")


@nox.session(python="3.12")
def style_guard(session: nox.Session) -> None:
    """Enforce the public-document style ban (no emojis, no em/en-dashes)."""
    _install(session)
    session.run("python", str(TOOLS / "lint_no_emoji.py"), "README.md", str(DOCS))
    session.run("python", str(TOOLS / "check_dashes.py"), "README.md", str(DOCS))


@nox.session(python="3.12")
def reproducibility(session: nox.Session) -> None:
    """Run the receipt replay regression tests."""
    _install(session, "test")
    session.run(
        "pytest",
        "tests/reproducibility",
        "-m",
        "reproducibility",
        *session.posargs,
    )


@nox.session(python="3.12")
def docs(session: nox.Session) -> None:
    """Strict mkdocs build."""
    _install(session, "docs")
    session.run("mkdocs", "build", "--strict")


@nox.session(python="3.12")
def docs_serve(session: nox.Session) -> None:
    """Local docs preview with autoreload."""
    _install(session, "docs")
    session.run("mkdocs", "serve", "--strict")


@nox.session(python="3.12")
def bench(session: nox.Session) -> None:
    """Quick airspeed-velocity microbenchmarks."""
    _install(session, "bench")
    session.run("asv", "run", "--quick", "HEAD^!", *session.posargs)


@nox.session(python="3.12")
def build(session: nox.Session) -> None:
    """Build sdist and wheel locally."""
    session.install("build>=1.2")
    out = ROOT / "dist"
    if out.exists():
        shutil.rmtree(out)
    session.run("python", "-m", "build", "--sdist", "--wheel")


@nox.session(python="3.12")
def clean(session: nox.Session) -> None:
    """Remove build, cache, and coverage artefacts."""
    targets = [
        "build", "dist", "wheelhouse", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", ".nox", ".coverage", "coverage.xml", "htmlcov", "site",
    ]
    for t in targets:
        path = ROOT / t
        if path.exists():
            session.log(f"removing {path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
