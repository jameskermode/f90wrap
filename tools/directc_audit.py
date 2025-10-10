#!/usr/bin/env python3
"""Audit Direct-C generation coverage across examples."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
REPORT_PATH = Path(__file__).resolve().parent.parent / "direct_c_coverage.md"
JSON_PATH = Path(__file__).resolve().parent.parent / "direct_c_coverage.json"
GFORT_FLAGS = ["-fPIC", "-xf95-cpp-input"]

@dataclass
class ExampleResult:
    name: str
    status: str  # PASS / FAIL
    note: str


def parse_makefile(makefile: Path) -> Tuple[str, List[str]]:
    name: str | None = None
    wrapflags: List[str] = ["-v"]  # default from make.meson.inc
    name_re = re.compile(r"^\s*NAME\s*[:+]?=\s*([^\s]+)")
    wrap_re = re.compile(r"WRAPFLAGS\s*\+=\s*(.+)")
    wrap_set_re = re.compile(r"^\s*WRAPFLAGS\s*:?=\s*(.+)")
    eval_re = re.compile(r"WRAPFLAGS\s*\+=\s*([^\)]+)")

    def add_tokens(text: str) -> None:
        tokens = [tok.strip('"') for tok in text.split()]
        tokens = [tok.rstrip(')') for tok in tokens]
        wrapflags.extend(tok for tok in tokens if tok)

    with makefile.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = name_re.match(line)
            if m and name is None:
                name = m.group(1)
                continue
            m2 = wrap_set_re.match(line)
            if m2:
                wrapflags = []
                add_tokens(m2.group(1))
                continue
            m3 = wrap_re.search(line)
            if m3:
                add_tokens(m3.group(1))
                continue
            if "WRAPFLAGS" in line and "WRAPFLAGS" in raw and "$(eval" in raw:
                m4 = eval_re.search(raw)
                if m4:
                    add_tokens(m4.group(1))
    if not name:
        name = makefile.parent.name.lower()
    return name, wrapflags


def collect_sources(example_dir: Path) -> List[Path]:
    chosen: dict[str, Path] = {}
    allowed_suffixes = {".f", ".f90", ".f95", ".fpp", ".F", ".F90"}
    for path in example_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix not in allowed_suffixes:
            continue
        if path.name.lower().startswith("f90wrap_"):
            continue
        stem = path.name.rsplit('.', 1)[0]
        key = stem.lower()
        previous = chosen.get(key)
        if previous is None or path.suffix.lower() == '.fpp':
            chosen[key] = path
    return sorted(chosen.values())


def preprocess_sources(workdir: Path, sources: List[Path]) -> List[Path]:
    generated: List[Path] = []
    for src in sources:
        if src.suffix.lower() == ".fpp":
            generated.append(src)
            continue
        out = workdir / (src.name + ".fpp")
        cmd = ["gfortran", "-E", *GFORT_FLAGS, src.name, "-o", out.name]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
        if result.returncode != 0:
            raise RuntimeError(f"Preprocess failed for {src.name}: {result.stderr.strip()}")
        generated.append(out)
    return generated


def run_direct_c(workdir: Path, module_name: str, wrapflags: List[str], fpp_files: List[Path]) -> subprocess.CompletedProcess[str]:
    entrypoint = Path(__file__).resolve().parent.parent / "f90wrap" / "scripts" / "main.py"
    cmd = [sys.executable, str(entrypoint), "--direct-c", "-m", module_name]
    cmd.extend(wrapflags)
    cmd.extend([p.name for p in fpp_files])
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parent.parent
    env['PYTHONPATH'] = str(repo_root) + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run(cmd, capture_output=True, text=True, cwd=workdir, env=env)


def audit_example(example_dir: Path) -> ExampleResult:
    makefile = example_dir / "Makefile.meson"
    if not makefile.exists():
        return ExampleResult(example_dir.name, "SKIP", "No Makefile.meson")

    module_name, wrapflags = parse_makefile(makefile)

    sources_orig = collect_sources(example_dir)
    if not sources_orig:
        return ExampleResult(example_dir.name, "SKIP", "No Fortran sources")

    with tempfile.TemporaryDirectory(prefix=f"directc_{example_dir.name}_") as tmp:
        workdir = Path(tmp) / example_dir.name
        shutil.copytree(example_dir, workdir)
        sources = collect_sources(workdir)
        try:
            fpp_files = preprocess_sources(workdir, sources)
            result = run_direct_c(workdir, module_name, wrapflags, fpp_files)
        except Exception as exc:  # preprocessing failure
            return ExampleResult(example_dir.name, "FAIL", str(exc))

        note = result.stderr.strip() or result.stdout.strip()
        if result.returncode == 0:
            status = "PASS"
            note = "Generation succeeded"
        else:
            status = "FAIL"
            if note:
                lines = note.splitlines()
                note = " ".join(lines[:2])
            else:
                note = "f90wrap returned non-zero"

    return ExampleResult(example_dir.name, status, note)


def write_reports(results: List[ExampleResult]) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = sum(1 for r in results if r.status != "SKIP")
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")

    lines = [
        "# Direct-C Coverage Report",
        "",
        f"Generated: {timestamp}",
        "",
        f"Total examples: {len(results)}",
        f"Direct-C PASS: {passed}",
        f"Direct-C FAIL: {failed}",
        f"Skipped: {skipped}",
        "",
        "| Example | Status | Notes |",
        "|---------|--------|-------|",
    ]
    for r in sorted(results, key=lambda x: (x.status != "PASS", x.name.lower())):
        note = r.note.replace("|", "/") if r.note else ""
        lines.append(f"| {r.name} | {r.status} | {note} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n")

    data = {
        "generated": timestamp,
        "summary": {
            "examples": len(results),
            "total_considered": total,
            "pass": passed,
            "fail": failed,
            "skip": skipped,
        },
        "results": [r.__dict__ for r in results],
    }
    JSON_PATH.write_text(json.dumps(data, indent=2))


def main() -> None:
    results: List[ExampleResult] = []
    for example_dir in sorted(EXAMPLES_DIR.iterdir()):
        if not example_dir.is_dir():
            continue
        if example_dir.name.startswith('.'):
            continue
        result = audit_example(example_dir)
        results.append(result)
        print(f"{example_dir.name}: {result.status} - {result.note}")

    write_reports(results)


if __name__ == "__main__":
    main()
