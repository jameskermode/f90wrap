import unittest

from pywrapper import m_test

class TestKindMap(unittest.TestCase):

    def test_real(self):
        _ = m_test.test_real(1.)

    def test_real4(self):
        _ = m_test.test_real4(2.)

    def test_real8(self):
        _ = m_test.test_real8(3.)

if __name__ == '__main__':
    unittest.main()
