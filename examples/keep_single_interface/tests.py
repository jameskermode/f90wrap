import unittest

from pywrapper import m_test

class TestInterface(unittest.TestCase):

    def test_interface(self):
        out = m_test.an_interface(1.0)
        assert out == 2.0

    def test_subroutine(self):
        out = m_test.a_subroutine(1.0)
        assert out == 2.0

if __name__ == '__main__':

    unittest.main()
