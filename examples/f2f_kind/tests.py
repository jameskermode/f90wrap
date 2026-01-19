import unittest
import numpy as np

from pywrapper import m_kind_test

class TestError(unittest.TestCase):

    def test_kind(self):
        an_int = np.array([0], dtype=np.int32)
        a_real = np.array([0], dtype=np.float32)
        m_kind_test.kind_test(an_int, a_real)
        self.assertEqual(an_int, 1)
        self.assertEqual(a_real, 1.0)

if __name__ == '__main__':

    unittest.main()
