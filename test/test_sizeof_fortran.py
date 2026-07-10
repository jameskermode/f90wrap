import os
import unittest

from f90wrap.sizeof_fortran import (
    probe_sizeof_fortran_t,
    resolve_sizeof_fortran_t,
)


class TestSizeofFortran(unittest.TestCase):

    def test_explicit_size_does_not_run_compiler(self):
        size = resolve_sizeof_fortran_t(
            explicit_size=10,
            compiler="compiler-that-does-not-exist",
        )

        self.assertEqual(size, 10)

    def test_explicit_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolve_sizeof_fortran_t(explicit_size=0)

    def test_probe_accepts_compiler_flags(self):
        compiler = os.environ.get("FC", "gfortran")

        size = probe_sizeof_fortran_t(f"{compiler} -fPIC")

        self.assertGreater(size, 0)

    def test_probe_reports_missing_compiler(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot run"):
            probe_sizeof_fortran_t("compiler-that-does-not-exist")


if __name__ == "__main__":
    unittest.main()
