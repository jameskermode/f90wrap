"""End-to-end test for Direct-C code generation."""

import unittest
import tempfile
import os
import subprocess
from pathlib import Path


class TestDirectCE2E(unittest.TestCase):
    """Test Direct-C wrapper generation end-to-end (no compilation)."""

    def setUp(self):
        """Create temporary directory for test artifacts."""
        self.temp_dir = tempfile.mkdtemp(prefix='f90wrap_directc_test_')
        self.orig_dir = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        os.chdir(self.orig_dir)
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_minimal_module_generation(self):
        """Test Direct-C generation for minimal Fortran module."""
        # Write minimal Fortran module
        fortran_code = """
module testmod
    implicit none
contains
    subroutine test_sub(n)
        integer, intent(in) :: n
        print *, "n =", n
    end subroutine test_sub

    function test_func(x) result(y)
        real, intent(in) :: x
        real :: y
        y = x * 2.0
    end function test_func
end module testmod
"""
        fortran_file = Path(self.temp_dir) / 'testmod.f90'
        fortran_file.write_text(fortran_code)

        # Run f90wrap with --direct-c flag
        result = subprocess.run(
            ['f90wrap', '--direct-c', 'testmod.f90'],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Check generation succeeded
        self.assertEqual(result.returncode, 0,
                        f"f90wrap failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

        # Check expected files were generated
        self.assertTrue((Path(self.temp_dir) / 'f90wrap_testmod.f90').exists(),
                       "Fortran helper not generated")
        # Python wrapper is generated as mod.py by default
        python_wrapper = Path(self.temp_dir) / 'mod.py'
        if not python_wrapper.exists():
            # List actual files for debugging
            files = list(Path(self.temp_dir).iterdir())
            self.fail(f"Python wrapper not found. Files: {[f.name for f in files]}")
        self.assertTrue((Path(self.temp_dir) / '_testmod.c').exists(),
                       "Direct-C extension not generated")

    def test_generated_c_structure(self):
        """Test generated C code has expected structure."""
        # Write minimal Fortran module
        fortran_code = """
module simple
    implicit none
contains
    subroutine greet(n)
        integer, intent(in) :: n
        print *, "Hello", n
    end subroutine greet
end module simple
"""
        fortran_file = Path(self.temp_dir) / 'simple.f90'
        fortran_file.write_text(fortran_code)

        # Generate wrappers
        result = subprocess.run(
            ['f90wrap', '--direct-c', 'simple.f90'],
            capture_output=True,
            text=True,
            timeout=30
        )
        self.assertEqual(result.returncode, 0)

        # Read generated C code
        c_file = Path(self.temp_dir) / '_simple.c'
        self.assertTrue(c_file.exists())
        c_code = c_file.read_text()

        # Check for essential C components
        self.assertIn('#include <Python.h>', c_code,
                     "Missing Python.h include")
        self.assertIn('#include <numpy/arrayobject.h>', c_code,
                     "Missing numpy include")
        self.assertIn('PyMODINIT_FUNC', c_code,
                     "Missing module init function")
        self.assertIn('PyInit_simple', c_code,
                     "Missing PyInit function")
        self.assertIn('PyMethodDef', c_code,
                     "Missing method table")
        self.assertIn('simple_methods', c_code,
                     "Missing methods array")

        # Note: Current implementation only wraps procedures with requires_helper=True
        # Simple scalar integer procedures are ISO C compatible (requires_helper=False)
        # So wrapper may be empty. This will be addressed in Phase 3.
        # For now, just verify the C structure is valid.

        # Check basic syntax validity
        self.assertEqual(c_code.count('{'), c_code.count('}'),
                        "Unbalanced braces in generated C code")

    def test_normal_mode_unchanged(self):
        """Test that normal mode (without --direct-c) still works."""
        # Write minimal Fortran module
        fortran_code = """
module normal
    implicit none
contains
    subroutine do_something(x)
        real, intent(in) :: x
        print *, x
    end subroutine do_something
end module normal
"""
        fortran_file = Path(self.temp_dir) / 'normal.f90'
        fortran_file.write_text(fortran_code)

        # Run f90wrap WITHOUT --direct-c flag
        result = subprocess.run(
            ['f90wrap', 'normal.f90'],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Check generation succeeded
        self.assertEqual(result.returncode, 0,
                        f"Normal f90wrap failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

        # Check standard files generated
        self.assertTrue((Path(self.temp_dir) / 'f90wrap_normal.f90').exists(),
                       "Fortran helper not generated in normal mode")
        # Python wrapper is mod.py by default
        self.assertTrue((Path(self.temp_dir) / 'mod.py').exists(),
                       "Python wrapper not generated in normal mode")

        # Check Direct-C file was NOT generated
        self.assertFalse((Path(self.temp_dir) / '_normal.c').exists(),
                        "Direct-C file should not be generated without --direct-c flag")


if __name__ == '__main__':
    unittest.main()
