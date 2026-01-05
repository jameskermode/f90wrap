import unittest

class TestImport(unittest.TestCase):
    def test_import(self):
        import _pywrapper_sign_f2py_meson

        # This document what seems to be a limitation in f2py
        with self.assertRaises((SystemError, ValueError)):
            _ = _pywrapper_sign_f2py_meson.string_in_array_optional()

if __name__ == '__main__':

    unittest.main()
