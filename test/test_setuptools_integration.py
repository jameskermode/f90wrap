"""Tests for setuptools integration (f90wrap/setuptools_ext.py)."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestSetuptoolsIntegration(unittest.TestCase):
    """Test F90WrapExtension setuptools integration."""

    def setUp(self):
        """Create temporary test package."""
        self.test_dir = tempfile.mkdtemp(prefix="test_f90wrap_setuptools_")
        self.addCleanup(shutil.rmtree, self.test_dir)

        # Create minimal Fortran source
        src_dir = Path(self.test_dir) / "src"
        src_dir.mkdir()

        fortran_src = src_dir / "testmod.f90"
        fortran_src.write_text("""
module testmod
    implicit none
contains
    subroutine add_two(a, b, result)
        real(8), intent(in) :: a, b
        real(8), intent(out) :: result
        result = a + b
    end subroutine add_two
end module testmod
""")

        # Create pyproject.toml
        pyproject = Path(self.test_dir) / "pyproject.toml"
        pyproject.write_text("""
[build-system]
requires = ["setuptools", "numpy", "f90wrap"]

[project]
name = "testpkg"
version = "0.1.0"

[tool.setuptools.packages]
find = {}
""")

        # Create setup.py
        setup_py = Path(self.test_dir) / "setup.py"
        setup_py.write_text("""
from setuptools import setup
from f90wrap.setuptools_ext import F90WrapExtension, build_ext_cmdclass

setup(
    ext_modules=[F90WrapExtension("testmod", ["src/testmod.f90"])],
    cmdclass=build_ext_cmdclass()
)
""")

    def test_build_ext_inplace(self):
        """Test build_ext --inplace creates package structure."""
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=self.test_dir,
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0, f"Build failed: {result.stderr}")

        # Check package structure created
        pkg_dir = Path(self.test_dir) / "testpkg"
        self.assertTrue(pkg_dir.exists(), "Package directory not created")
        self.assertTrue((pkg_dir / "__init__.py").exists(), "__init__.py missing")
        self.assertTrue((pkg_dir / "testmod.py").exists(), "testmod.py missing")

        # Check extension module exists with platform suffix
        so_files = list(pkg_dir.glob("_testmod*.so"))
        self.assertTrue(so_files, "Extension .so file not found")

    def test_pip_install(self):
        """Test pip install creates installable package."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation",
             "--no-deps", self.test_dir],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": ""}
        )

        self.assertEqual(result.returncode, 0, f"Install failed: {result.stderr}")

        # Try to import
        import_result = subprocess.run(
            [sys.executable, "-c", "import testpkg; print('SUCCESS')"],
            capture_output=True,
            text=True
        )

        self.assertEqual(import_result.returncode, 0,
                        f"Import failed: {import_result.stderr}")
        self.assertIn("SUCCESS", import_result.stdout)

        # Cleanup: uninstall
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "testpkg"],
                      capture_output=True)

    def test_package_structure_correctness(self):
        """Test that package structure matches documentation."""
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=self.test_dir,
            capture_output=True
        )

        pkg_dir = Path(self.test_dir) / "testpkg"
        init_content = (pkg_dir / "__init__.py").read_text()

        # Check __init__.py imports from module
        self.assertIn("from .testmod import *", init_content)

        # Check testmod.py uses relative imports
        testmod_content = (pkg_dir / "testmod.py").read_text()
        self.assertIn("from . import _testmod", testmod_content)
        # Should NOT have standalone "import _testmod" (must be relative)
        lines = testmod_content.split('\n')
        for line in lines:
            if line.strip() == "import _testmod":
                self.fail("Found absolute 'import _testmod', should be relative")

    def test_build_hyphenated_distribution(self):
        """Hyphenated distribution names should be normalised for packages."""
        pyproject = Path(self.test_dir) / "pyproject.toml"
        original = pyproject.read_text()
        pyproject.write_text(original.replace('name = "testpkg"', 'name = "test-pkg"'))
        self.addCleanup(pyproject.write_text, original)

        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=self.test_dir,
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0, f"Build failed: {result.stderr}")

        pkg_dir = Path(self.test_dir) / "test_pkg"
        self.assertTrue(pkg_dir.exists(), "Sanitised package directory missing")
        self.assertTrue((pkg_dir / "testmod.py").exists(), "Module not copied into sanitised package")
        so_files = list(pkg_dir.glob("_testmod*.so"))
        self.assertTrue(so_files, "Shared object missing in sanitised package")


if __name__ == '__main__':
    unittest.main()
