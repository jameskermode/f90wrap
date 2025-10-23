import unittest

import _pywrapper_sign_f2py_distutils

class TestWithSignature(unittest.TestCase):

    # This document what seems to be a limitation in f2py
    def test_string_in_array_optional_not_present(self):
        with self.assertRaises((SystemError, ValueError)):
            _ = _pywrapper_sign_f2py_distutils.string_in_array_optional()

if __name__ == '__main__':

    unittest.main()
