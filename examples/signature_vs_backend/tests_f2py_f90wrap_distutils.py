import unittest

import _pywrapper_sign_f2py_f90wrap_distutils

class TestWithSignature(unittest.TestCase):

    # This document what seems to be a limitation in f2py but not in f2py-f90wrap
    def test_string_in_array_optional_not_present(self):
        output = _pywrapper_sign_f2py_f90wrap_distutils.string_in_array_optional()
        self.assertEqual(output, 2)

if __name__ == '__main__':

    unittest.main()
