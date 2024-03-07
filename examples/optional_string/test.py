import unittest
import numpy as np
from packaging import version

from pywrapper import m_string_test

class TestTypeCheck(unittest.TestCase):

    def test_string_in_1(self):
        m_string_test.string_in('yo')

    def test_string_in_2(self):
        m_string_test.string_in(np.unicode_('yo'))

    def test_string_in_3(self):
        m_string_test.string_in(np.string_('yo'))

    def test_string_to_string(self):
        in_string = 'yo'
        out_string = m_string_test.string_to_string(in_string)
        self.assertEqual(in_string, out_string.decode("utf-8").strip())

    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.24.0") , "This test is known to fail on numpy version older than 1.24.0")
    def test_string_to_string_array(self):
        in_string = np.array(['first ', 'second'], dtype='S6')
        out_string = np.array([' '*6, ' '*6], dtype='S6')
        m_string_test.string_to_string_array(in_string, out_string)
        for i in range(in_string.size):
            self.assertEqual(in_string[i], out_string[i])

    def test_string_out(self):
        out_string = m_string_test.string_out()
        self.assertEqual(out_string.decode("utf-8").strip(), "output string")

    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.23.5") , "This test is known to fail on numpy version older than 1.23.5")
    def test_string_out_optional(self):
        out_string = np.array(' '*13, dtype='S13')
        m_string_test.string_out_optional(out_string)
        self.assertEqual(out_string, np.array("output string", dtype='S13'))

    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.26.3") , "Bug solved in https://github.com/numpy/numpy/pull/24791")
    def test_string_out_optional_2(self):
        m_string_test.string_out_optional()

    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.24.0") , "This test is known to fail on numpy version older than 1.24.0")
    def test_string_out_optional_array(self):
        out_string_array = np.array([' '*13, ' '*13], dtype='S13')
        m_string_test.string_out_optional_array(out_string_array)
        for out_string in out_string_array:
            self.assertEqual(out_string, np.array("output string", dtype='S13'))

    def test_string_out_optional_array_2(self):
        m_string_test.string_out_optional_array()

if __name__ == '__main__':

    unittest.main()
