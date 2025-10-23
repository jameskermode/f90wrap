import unittest
import numpy as np
from packaging import version

try:
    import _pywrapper_sign_f2py_f90wrap_distutils as _pywrapper_sign
except ImportError:
    pass

try:
    import _pywrapper_sign_f2py_f90wrap_meson as _pywrapper_sign
except ImportError:
    pass

try:
    import _pywrapper_sign_f2py_distutils as _pywrapper_sign
except ImportError:
    pass

try:
    import _pywrapper_sign_f2py_meson as _pywrapper_sign
except ImportError:
    pass


class TestWithSignature(unittest.TestCase):

    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.24.0") , "f2py bug solved https://github.com/numpy/numpy/issues/24706")
    def test_string_in_array(self):
        in_array = np.array(['one', 'two'], dtype='S3')
        output = _pywrapper_sign.string_in_array(in_array)
        self.assertEqual(output, 0)

    @unittest.skipIf(version.parse(np.version.version) < version.parse("1.24.0") , "f2py bug solved https://github.com/numpy/numpy/issues/24706")
    def test_string_in_array_optional_present(self):
        in_array = np.array(['one', 'two'], dtype='S3')
        output = _pywrapper_sign.string_in_array_optional(in_array)
        self.assertEqual(output, 0)

if __name__ == '__main__':

    unittest.main()
