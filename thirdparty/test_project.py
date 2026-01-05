#!/usr/bin/env python3
"""
Run third-party integration checks for f90wrap.

This script clones external repositories into a temporary directory, patches
their build metadata to prefer the locally-installed f90wrap, then runs build,
import, and optional test commands defined in `thirdparty/projects.yaml`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ProjectConfig:
    name: str
    repo: str
    branch: str = ""
    build_cmd: str = "pip install . --no-build-isolation"
    import_test: str = ""
    test_cmd: str = ""
    env: dict = field(default_factory=dict)
    timeout: int = 600
    enabled: bool = True


def _run(cmd: str, cwd: Path, timeout: int, env: Optional[dict] = None) -> tuple[bool, str, float]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
        )
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s: {cmd}", time.time() - start

    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out, time.time() - start


def _clone(repo: str, dest: Path, branch: str) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest)]
    subprocess.run(cmd, check=True)


def _run_import(import_stmt: str) -> tuple[bool, str]:
    cmd = f'python -c "{import_stmt}"'
    try:
        # Run from /tmp to avoid importing local f90wrap source instead of installed package
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60, cwd="/tmp"
        )
    except Exception as e:
        return False, str(e)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()


def load_projects(config_path: Path) -> dict[str, ProjectConfig]:
    data = yaml.safe_load(config_path.read_text())
    configs: dict[str, ProjectConfig] = {}
    for proj in data.get("projects", []):
        repo = proj.get("repo", "")
        name = repo.split("/")[-1] if "/" in repo else repo
        configs[name] = ProjectConfig(
            name=name,
            repo=repo,
            branch=proj.get("branch", "") or "",
            build_cmd=proj.get("build_cmd", "pip install . --no-build-isolation"),
            import_test=proj.get("import_test", "") or "",
            test_cmd=proj.get("test_cmd", "") or "",
            timeout=int(proj.get("timeout", 600)),
            enabled=bool(proj.get("enabled", True)),
        )
    return configs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("projects", nargs="*", help="Project names (default: enabled ones)")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("projects.yaml"))
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("F90WRAP_THIRDPARTY_ROOT", "/tmp/f90wrap-thirdparty")))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    configs = load_projects(args.config)
    selected = args.projects or [name for name, cfg in configs.items() if cfg.enabled]
    if not selected:
        print("No projects selected.")
        return 0

    overall_ok = True
    for name in selected:
        cfg = configs.get(name)
        if cfg is None:
            print(f"Unknown project: {name}")
            overall_ok = False
            continue

        project_dir = args.root / "repos" / cfg.name
        print("\n" + "=" * 60)
        print(f"Third-party: {cfg.name} ({cfg.repo})")
        print("=" * 60)

        try:
            _clone(cfg.repo, project_dir, cfg.branch)
        except subprocess.CalledProcessError as e:
            print(f"Clone FAILED: {e}")
            overall_ok = False
            continue

        # Patch metadata to prefer the locally-installed f90wrap
        subprocess.run(
            ["python", str(Path(__file__).with_name("patch_project.py")), str(project_dir)],
            check=False,
            cwd=str(project_dir),
        )

        print(f"Build: {cfg.build_cmd}")
        ok, out, elapsed = _run(cfg.build_cmd, project_dir, cfg.timeout, cfg.env)
        if args.verbose or not ok:
            print(out[-4000:] if len(out) > 4000 else out)
        print(f"Build {'OK' if ok else 'FAIL'} in {elapsed:.1f}s")
        if not ok:
            overall_ok = False
            continue

        if cfg.import_test:
            print(f"Import: {cfg.import_test}")
            imp_ok, imp_out = _run_import(cfg.import_test)
            if args.verbose or not imp_ok:
                print(imp_out)
            print(f"Import {'OK' if imp_ok else 'FAIL'}")
            if not imp_ok:
                overall_ok = False

        if cfg.test_cmd and not args.skip_tests:
            print(f"Test: {cfg.test_cmd}")
            test_ok, test_out, test_elapsed = _run(cfg.test_cmd, project_dir, cfg.timeout, cfg.env)
            if args.verbose or not test_ok:
                print(test_out[-4000:] if len(test_out) > 4000 else test_out)
            print(f"Test {'OK' if test_ok else 'FAIL'} in {test_elapsed:.1f}s")
            if not test_ok:
                overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

