#!/usr/bin/env python3
"""
Patch third-party projects to prefer the locally-installed f90wrap.

In CI we install f90wrap from the PR branch first, then run third-party builds
with --no-build-isolation so they use that environment.

This patcher removes explicit f90wrap build dependencies where possible to
avoid pip pulling a different f90wrap from PyPI.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def patch_pyproject_toml(filepath: Path) -> bool:
    if not filepath.exists():
        return False

    content = filepath.read_text()
    if "f90wrap" not in content:
        return False

    new_content = re.sub(
        r'["\']f90wrap[^"\']*["\'],?\s*',
        "",
        content,
    )

    if new_content == content:
        return False

    filepath.write_text(new_content)
    return True


def patch_setup_py(filepath: Path) -> bool:
    if not filepath.exists():
        return False

    content = filepath.read_text()
    if "f90wrap" not in content:
        return False

    new_content = re.sub(
        r'["\']f90wrap[^"\']*["\'],?\s*',
        "",
        content,
    )

    if new_content == content:
        return False

    filepath.write_text(new_content)
    return True


def patch_requirements(filepath: Path) -> bool:
    if not filepath.exists():
        return False

    lines = filepath.read_text().splitlines()
    new_lines = [line for line in lines if not line.strip().startswith("f90wrap")]
    if len(new_lines) == len(lines):
        return False

    filepath.write_text("\n".join(new_lines) + "\n")
    return True


def patch_project(project_dir: Path) -> bool:
    patched = False
    patched |= patch_pyproject_toml(project_dir / "pyproject.toml")
    patched |= patch_setup_py(project_dir / "setup.py")
    patched |= patch_requirements(project_dir / "requirements.txt")
    patched |= patch_requirements(project_dir / "requirements-dev.txt")
    return patched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir", type=Path)
    args = ap.parse_args()

    changed = patch_project(args.project_dir)
    return 0 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())

