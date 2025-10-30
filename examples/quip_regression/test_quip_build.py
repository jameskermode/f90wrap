#!/usr/bin/env python
"""
QUIP Regression Test for f90wrap

This test checks that f90wrap can successfully build QUIP/quippy,
a real-world project that uses f90wrap extensively. This helps
prevent regressions in f90wrap that would break QUIP.

NOTE: This test requires OpenBLAS to be installed and pkg-config
to be able to find it. The test will be skipped if these dependencies
are not available.
"""

import os
import sys
import subprocess
import shutil
import tempfile
import unittest


class TestQUIPBuild(unittest.TestCase):
    """Test that QUIP can be built with the current version of f90wrap"""

    QUIP_REPO = "https://github.com/libAtoms/QUIP.git"
    QUIP_BRANCH = "mesonify"

    @classmethod
    def _check_dependencies(cls):
        """Check if required dependencies are available"""
        # Check for pkg-config
        try:
            subprocess.run(["pkg-config", "--version"], check=True,
                         capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "pkg-config not found"

        # Check for openblas via pkg-config
        try:
            subprocess.run(["pkg-config", "--exists", "openblas"],
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return False, "OpenBLAS not found via pkg-config"

        return True, None

    @classmethod
    def setUpClass(cls):
        """Clone and build QUIP once for all tests"""
        # Check dependencies first
        deps_ok, reason = cls._check_dependencies()
        if not deps_ok:
            raise unittest.SkipTest(f"QUIP regression test skipped: {reason}")

        # Install f90wrap from local source (non-editable to avoid meson-python issues)
        print("\nInstalling f90wrap from local source...")
        f90wrap_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", f90wrap_root, "--force-reinstall"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install f90wrap:\\nSTDOUT:\\n{result.stdout}\\n\\nSTDERR:\\n{result.stderr}")

        cls.test_dir = tempfile.mkdtemp(prefix="f90wrap_quip_test_")
        cls.quip_dir = os.path.join(cls.test_dir, "QUIP")

        print(f"\nCloning QUIP to {cls.quip_dir}...")
        subprocess.run(
            ["git", "clone", "--recursive", "--branch", cls.QUIP_BRANCH,
             cls.QUIP_REPO, cls.quip_dir],
            check=True,
            capture_output=True
        )

        # Patch QUIP's pyproject.toml to use local f90wrap instead of git master
        # This ensures we test the current branch, not master
        print("Patching QUIP to use local f90wrap...")
        f90wrap_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        quippy_pyproject = os.path.join(cls.quip_dir, "quippy", "pyproject.toml")

        with open(quippy_pyproject, "r") as f:
            content = f.read()

        # Replace git URL with local path
        content = content.replace(
            'f90wrap @ git+https://github.com/jameskermode/f90wrap.git@master',
            f'f90wrap @ file://{f90wrap_root}'
        )

        with open(quippy_pyproject, "w") as f:
            f.write(content)

        print("Building QUIP libraries...")
        result = subprocess.run(
            ["meson", "setup", "builddir"],
            cwd=cls.quip_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Meson setup failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

        result = subprocess.run(
            ["meson", "compile", "-C", "builddir"],
            cwd=cls.quip_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Meson compile failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory"""
        if hasattr(cls, 'test_dir') and os.path.exists(cls.test_dir):
            print(f"\nCleaning up {cls.test_dir}...")
            shutil.rmtree(cls.test_dir)

    def test_01_quippy_package_builds(self):
        """Test that quippy Python package can be built with f90wrap"""
        print("\nBuilding quippy Python package...")

        quippy_dir = os.path.join(self.quip_dir, "quippy")

        # Build a wheel first to avoid editable install issues
        # Use --no-build-isolation to ensure it uses our local f90wrap
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-build-isolation", "-v",
             "--no-deps", "-w", "dist"],
            cwd=quippy_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            self.fail(f"quippy wheel build failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

        # Find the built wheel
        dist_dir = os.path.join(quippy_dir, "dist")
        wheels = [f for f in os.listdir(dist_dir) if f.endswith('.whl')]
        self.assertTrue(len(wheels) > 0, "No wheel file was created")
        wheel_path = os.path.join(dist_dir, wheels[0])

        # Install the wheel (non-editable)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", wheel_path, "--force-reinstall"],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0,
                        f"quippy install failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

    def test_02_quippy_imports(self):
        """Test that quippy can be imported"""
        print("\nTesting quippy import...")

        result = subprocess.run(
            [sys.executable, "-c", "import quippy; print('Import successful')"],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0,
                        f"quippy import failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
        self.assertIn("Import successful", result.stdout)

    def test_03_quippy_basic_usage(self):
        """Test basic quippy functionality"""
        print("\nTesting basic quippy usage...")

        test_code = """
import quippy
import ase

# Create a simple atoms object
at = ase.Atoms('H', positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)

# Convert to QUIP format
at_quip = quippy.convert.ase_to_quip(at)

# Test basic properties
assert at_quip.n == 1
assert at_quip.z[0] == 1
print('Basic usage test passed')
"""

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0,
                        f"quippy basic usage failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
        self.assertIn("Basic usage test passed", result.stdout)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
