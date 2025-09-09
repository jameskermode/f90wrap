import unittest

from pywrapper import m_test

class TestReturnBool(unittest.TestCase):

    def test_return_logical(self):
        assert(m_test.return_logical(True) == True)
        assert(isinstance(m_test.return_logical(True), bool))
        assert(m_test.return_logical(False) == False)
        assert(isinstance(m_test.return_logical(False), bool))

if __name__ == '__main__':

    unittest.main()
