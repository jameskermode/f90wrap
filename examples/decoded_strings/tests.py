import unittest

from pywrapper import m_test

class TestDecodedString(unittest.TestCase):

    def test_return_string(self):
        string_out = m_test.return_string()
        assert(string_out == '-_-::this is a string with ASCII, / and 123...::-_-')
        assert(isinstance(string_out, str))

    def test_func_return_string(self):
        string_out = m_test.func_return_string()
        assert(string_out == '-_-::this is a string with ASCII, / and 123...::-_-')
        assert(isinstance(string_out, str))

if __name__ == '__main__':

    unittest.main()
