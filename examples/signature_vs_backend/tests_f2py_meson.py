import unittest

class TestImport(unittest.TestCase):
    def test_import(self):
        # This document what seems to be a limitation in f2py
        with self.assertRaises(ImportError):
            import _pywrapper_sign_f2py_meson

if __name__ == '__main__':

    unittest.main()
