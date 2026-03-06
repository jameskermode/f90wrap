import unittest
import numpy as np
from pywrapper import m_test

class TestBoolSupport(unittest.TestCase):
    def setUp(self):
        self.obj = m_test.t_bool_wrapper()
        self.scalar_val = True
        self.static_array_val = np.array([True, True, False, True, True, False], dtype=bool)
        self.dynamic_array_val = np.array([False, True, True, False, True, True], dtype=bool)
        self.obj.init(self.scalar_val, self.static_array_val, self.dynamic_array_val)

    def tearDown(self):
        self.obj.free()

    def test_explicit_scalar_accessor(self):
        res_scalar = self.obj.get_scalar()
        self.assertEqual(res_scalar, self.scalar_val)

    def test_explicit_static_array_accessor(self):
        res_static = self.obj.get_static_array()
        np.testing.assert_array_equal(res_static, self.static_array_val)

    def test_explicit_dynamic_array_accessor(self):
        res_dynamic = np.array([False, False, False, False, False, False], dtype=bool)
        self.obj.get_dynamic_array(res_dynamic)
        np.testing.assert_array_equal(res_dynamic, self.dynamic_array_val)

    def test_implicit_scalar_accessor(self):
        res_scalar = self.obj.scalar
        self.assertEqual(res_scalar, self.scalar_val)

    def test_implicit_static_array_accessor(self):
        res_static = self.obj.static_array
        np.testing.assert_array_equal(res_static, self.static_array_val)

    def test_implicit_dynamic_array_accessor(self):
        res_dynamic = self.obj.dynamic_array
        np.testing.assert_array_equal(res_dynamic, self.dynamic_array_val)

    def test_implicit_scalar_setter(self):
        new_val = False
        self.obj.scalar = new_val
        self.assertEqual(self.obj.get_scalar(), new_val)

    def test_implicit_static_array_setter(self):
        new_val = np.array([False, False, True, False, False, True], dtype=bool)
        self.obj.static_array = new_val
        np.testing.assert_array_equal(self.obj.get_static_array(), new_val)

    def test_implicit_dynamic_array_setter(self):
        new_val = np.array([True, False, False, True, False, False], dtype=bool)
        self.obj.dynamic_array = new_val
        np.testing.assert_array_equal(self.obj.dynamic_array, new_val)

if __name__ == '__main__':
    unittest.main()
