import unittest
import numpy as np
from packaging import version

import _pywrapper_no_sign

class TestWithoutSignature(unittest.TestCase):

    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.24.0") , "This test is known to fail on numpy version older than 1.24.0, dtype=S# does not work")
    def test_string_in_array(self):
        in_array = np.array(['one', 'two'], dtype='S3')
        output = _pywrapper_no_sign.string_in_array(in_array)
        self.assertEqual(output, 0)

    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.24.0") , "This test is known to fail on numpy version older than 1.24.0, dtype=S# does not work")
    def test_string_in_array_optional_present(self):
        in_array = np.array(['one', 'two'], dtype='S3')
        output = _pywrapper_no_sign.string_in_array_optional(in_array)
        self.assertEqual(output, 0)

    def test_string_in_array_optional_not_present(self):
        with self.assertRaises((SystemError, ValueError)):
            _ = _pywrapper_no_sign.string_in_array_optional()

class TestWithoutSignatureWithCDtype(unittest.TestCase):

    @unittest.skipIf(version.parse(np.version.version) > version.parse("1.23.5") , "This test is known to fail on numpy version newer than 1.23.5, dtype=c should not be used")
    def test_string_in_array(self):
        in_array = np.array(['one', 'two'], dtype='c')
        output = _pywrapper_no_sign.string_in_array(in_array)
        self.assertEqual(output, 0)

    @unittest.skipIf(version.parse(np.version.version) > version.parse("1.23.5") , "This test is known to fail on numpy version newer than 1.23.5, dtype=c should not be used")
    def test_string_in_array_optional_present(self):
        in_array = np.array(['one', 'two'], dtype='c')
        output = _pywrapper_no_sign.string_in_array_optional(in_array)
        self.assertEqual(output, 0)

    def test_string_in_array_optional_not_present(self):
        with self.assertRaises((SystemError, ValueError)):
            _ = _pywrapper_no_sign.string_in_array_optional()

if __name__ == '__main__':

    unittest.main()
