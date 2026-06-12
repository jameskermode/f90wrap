import unittest

import numpy as np

from pywrapper import m_optional


def as_text(value):
    """Character returns come back as bytes (direct-c) or str (f2py)."""
    if isinstance(value, bytes):
        value = value.decode()
    return value.strip()


class TestOptionalPresent(unittest.TestCase):
    def test_scalar_present(self):
        self.assertEqual(m_optional.scalar_default(n=5), 5.0)

    def test_scalar_absent_uses_default(self):
        # Bug #369: direct-c saw present(n) == .true. with value 0.
        self.assertEqual(m_optional.scalar_default(), 42.0)

    def test_char_present(self):
        self.assertEqual(as_text(m_optional.char_default(name="hello")), "hello")

    def test_char_absent_uses_default(self):
        self.assertEqual(as_text(m_optional.char_default()), "ABSENT")

    def test_array_present(self):
        self.assertEqual(
            m_optional.array_default(v=np.array([1.0, 2.0, 3.0], dtype=np.float32)), 3
        )

    def test_array_absent(self):
        self.assertEqual(m_optional.array_default(), -1)


if __name__ == "__main__":
    unittest.main()
