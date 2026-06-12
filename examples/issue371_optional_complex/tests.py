import unittest

from pywrapper import m_complex


class TestOptionalComplex(unittest.TestCase):
    def test_present_single(self):
        # Previously impossible: the optional complex argument was dropped
        # entirely in transform.py before it reached either backend.
        self.assertEqual(m_complex.opt_complex(z=complex(2, 3)), 5)

    def test_present_negativeimag(self):
        self.assertEqual(m_complex.opt_complex(z=complex(4, -1)), 3)


if __name__ == "__main__":
    unittest.main()
