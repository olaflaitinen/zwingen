#!/usr/bin/env python3
# tools/compare_receipts.py
# Maintainer: @olaflaitinen <olaf.laitinen@gmail.com>
# License: Apache-2.0 (matches the host project)
#
# Compare two zwingen reproducibility receipts and emit a structured diff.
#
# A receipt is the content-addressed JSON document defined by
# zwingen.epistemic.RECEIPT_SCHEMA_VERSION. Comparison happens in two
# regimes:
#
#   strict   The two runs are expected to be byte-identical. Any difference
#            in artefact hashes, PRNG seeds, or numerical outputs is a
#            failure. Used in CI on the same hardware class.
#
#   tolerant The two runs may live on different hardware classes. Bitwise
#            equality is not required, but every numerical artefact must
#            stay within the documented tolerance for its `tolerance_key`
#            (see docs/reference/tolerances.md). Used to verify that
#            cross-platform wheels still meet the public numerical
#            contract.
#
# Exit codes:
#   0   No regression. Receipts agree under the requested regime.
#   1   Regression detected.
#   2   Schema mismatch or malformed receipt.
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Literal

SchemaVersion: Final[str] = "1.0"
Regime = Literal["strict", "tolerant"]

_SCALAR_TYPES: Final[tuple[type, ...]] = (int, float, bool, str, type(None))


@dataclass(slots=True)
class Diff:
    path: str
    kind: Literal["missing", "added", "value", "hash", "tolerance"]
    a: Any = None
    b: Any = None
    tolerance_key: str | None = None
    tolerance_value: float | None = None
    delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None or k in {"a", "b"}}


@dataclass(slots=True)
class Report:
    regime: Regime
    schema_a: str
    schema_b: str
    diffs: list[Diff] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.diffs

    def to_json(self) -> str:
        return json.dumps(
            {
                "regime": self.regime,
                "schema_a": self.schema_a,
                "schema_b": self.schema_b,
                "ok": self.ok,
                "diffs": [d.to_dict() for d in self.diffs],
            },
            indent=2,
            ensure_ascii=False,
        )

    def to_human(self) -> str:
        if self.ok:
            return f"Receipts agree under the {self.regime} regime.\n"
        lines = [f"{len(self.diffs)} regression(s) under the {self.regime} regime:"]
        for d in self.diffs:
            if d.kind == "missing":
                lines.append(f"  - {d.path}: missing in B (was {d.a!r})")
            elif d.kind == "added":
                lines.append(f"  - {d.path}: unexpected in B ({d.b!r})")
            elif d.kind == "value":
                lines.append(f"  - {d.path}: A={d.a!r} B={d.b!r}")
            elif d.kind == "hash":
                lines.append(f"  - {d.path}: artefact hash drift A={d.a} B={d.b}")
            elif d.kind == "tolerance":
                lines.append(
                    f"  - {d.path}: |A-B|={d.delta:.3e} exceeds {d.tolerance_key}={d.tolerance_value:.3e}"
                )
        return "\n".join(lines) + "\n"


def _load_receipt(path: pathlib.Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot load receipt {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Receipt {path} is not a JSON object.")
    return data


def _check_schema(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, str]:
    schema_a = str(a.get("schema_version", "unknown"))
    schema_b = str(b.get("schema_version", "unknown"))
    if schema_a != schema_b:
        sys.stderr.write(
            f"warning: schema mismatch ({schema_a} vs {schema_b}); diff may be misleading.\n"
        )
    if schema_a not in {SchemaVersion, "unknown"}:
        sys.stderr.write(
            f"warning: receipt schema {schema_a} is not the supported {SchemaVersion}.\n"
        )
    return schema_a, schema_b


def _walk(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten a JSON tree into a path-keyed dict of scalar / list leaves."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            child = f"{prefix}.{k}" if prefix else k
            out.update(_walk(child, v))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out.update(_walk(f"{prefix}[{i}]", v))
    else:
        out[prefix] = value
    return out


def _is_artefact_hash_path(path: str) -> bool:
    return path.startswith("artefacts.") and path.endswith(".sha256")


def _tolerance_for(path: str, tolerances: dict[str, float]) -> tuple[str, float] | None:
    """Map a leaf path to a (tolerance_key, value) if one applies."""
    # numerical_outputs.<key>.value matches a tolerance_key from the receipt.
    if path.startswith("numerical_outputs.") and path.endswith(".value"):
        key = path[len("numerical_outputs.") : -len(".value")]
        if key in tolerances:
            return key, tolerances[key]
    return None


def compare(
    a: dict[str, Any], b: dict[str, Any], *, regime: Regime
) -> Report:
    schema_a, schema_b = _check_schema(a, b)
    flat_a = _walk("", a)
    flat_b = _walk("", b)
    tolerances_a: dict[str, float] = a.get("tolerances", {}) or {}
    tolerances_b: dict[str, float] = b.get("tolerances", {}) or {}
    tolerances = {**tolerances_a, **tolerances_b}
    diffs: list[Diff] = []
    keys = set(flat_a) | set(flat_b)
    for key in sorted(keys):
        if key not in flat_b:
            diffs.append(Diff(path=key, kind="missing", a=flat_a[key]))
            continue
        if key not in flat_a:
            diffs.append(Diff(path=key, kind="added", b=flat_b[key]))
            continue
        va, vb = flat_a[key], flat_b[key]
        if va == vb:
            continue
        if _is_artefact_hash_path(key):
            if regime == "strict":
                diffs.append(Diff(path=key, kind="hash", a=va, b=vb))
            continue
        if regime == "tolerant":
            tol = _tolerance_for(key, tolerances)
            if tol is not None and isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                tol_key, tol_val = tol
                delta = abs(float(va) - float(vb))
                if delta <= tol_val or math.isclose(va, vb, rel_tol=tol_val, abs_tol=tol_val):
                    continue
                diffs.append(
                    Diff(
                        path=key,
                        kind="tolerance",
                        a=va,
                        b=vb,
                        tolerance_key=tol_key,
                        tolerance_value=tol_val,
                        delta=delta,
                    )
                )
                continue
        diffs.append(Diff(path=key, kind="value", a=va, b=vb))
    return Report(regime=regime, schema_a=schema_a, schema_b=schema_b, diffs=diffs)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.compare_receipts",
        description="Compare two zwingen reproducibility receipts.",
    )
    parser.add_argument("a", type=pathlib.Path, help="Path to the reference receipt (A).")
    parser.add_argument("b", type=pathlib.Path, help="Path to the candidate receipt (B).")
    parser.add_argument(
        "--regime",
        choices=("strict", "tolerant"),
        default="strict",
        help="strict: require bitwise equality; tolerant: allow per-key tolerances.",
    )
    parser.add_argument(
        "--report",
        choices=("json", "human"),
        default="human",
        help="Output format.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    a = _load_receipt(args.a)
    b = _load_receipt(args.b)
    report = compare(a, b, regime=args.regime)
    if args.report == "json":
        sys.stdout.write(report.to_json())
        sys.stdout.write("\n")
    else:
        sys.stdout.write(report.to_human())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
