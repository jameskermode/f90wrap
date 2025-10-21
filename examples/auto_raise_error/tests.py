import unittest
import numpy as np

from pywrapper import m_error

class TestError(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(RuntimeError) as context:
            m_error.auto_raise()
        self.assertEqual(str(context.exception).strip(), 'auto raise error')

    def test_raise_optional(self):
        with self.assertRaises(RuntimeError) as context:
            m_error.auto_raise_optional()
        self.assertEqual(str(context.exception).strip(), 'auto raise error optional')

    def test_raise_out(self):
        with self.assertRaises(RuntimeError) as context:
            m_error.auto_raise()
        self.assertEqual(str(context.exception).strip(), 'auto raise error')

    def test_no_raise(self):
        m_error.auto_no_raise()
        # Check that Error handling argument are correctly removed from interface
        with self.assertRaises(TypeError):
            ierr, errmsg = m_error.auto_no_raise()

    def test_no_raise_optional(self):
        m_error.auto_no_raise_optional()
        # Check that Error handling argument are correctly removed from interface
        with self.assertRaises(TypeError):
            ierr=1
            errmsg='error'
            m_error.auto_no_raise_optional(ierr, errmsg)

    def test_str_input(self):
        keyword='foo'
        m_error.str_input(keyword)
        m_error.str_input()

    def test_no_error_var(self):
        a_number, a_string = m_error.no_error_var()
        self.assertEqual(a_number, 1)
        self.assertEqual(a_string, b'a string')

if __name__ == '__main__':

    unittest.main()
