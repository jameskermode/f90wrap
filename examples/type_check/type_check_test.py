import unittest
import numpy as np
from packaging import version
import os

from pywrapper import m_type_test

class TestTypeCheck(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTypeCheck, self).__init__(*args, **kwargs)
        self._circle = m_type_test.t_circle()
        self._square = m_type_test.t_square()

    def test_derived_type_selection(self):
        out_circle = np.array([-1], dtype=np.int32)
        out_square = np.array([-1], dtype=np.int32)

        m_type_test.is_circle(self._circle, out_circle)
        m_type_test.is_circle(self._square, out_square)

        self.assertEqual(out_circle[0], 1)
        self.assertEqual(out_square[0], 0)

    def test_shape_selection_0d(self):
        out = np.array(-1, dtype=np.int32)
        m_type_test.write_array(out)

        self.assertEqual(out, 10)

    def test_kind_selection_int_0d(self):
        out = np.array(-1, dtype=np.int64)
        m_type_test.write_array(out)

        self.assertEqual(out, 11)

    def test_kind_selection_real32_0d(self):
        out = np.array(-1, dtype=np.float32)
        m_type_test.write_array(out)

        self.assertEqual(out, 12)

    def test_kind_selection_real64_0d(self):
        out = np.array(-1, dtype=np.float64)
        m_type_test.write_array(out)

        self.assertEqual(out, 13)

    def test_shape_selection_1d(self):
        out = np.array([-1], dtype=np.int32)
        m_type_test.write_array(out)

        self.assertEqual(out[0], 1)

    def test_shape_selection_2d(self):
        out = np.array([[-1]], dtype=np.int32)
        m_type_test.write_array(out)

        self.assertEqual(out[0][0], 2)

    def test_type_selection(self):
        out = np.array([-1], dtype=np.float32)
        m_type_test.write_array(out)

        self.assertEqual(out[0], 3)

    def test_kind_selection_float_1d(self):
        out = np.array([-1], dtype=np.float64)
        m_type_test.write_array(out)

        self.assertEqual(out[0], 4)

    @unittest.skip("Bool are not supported in interfaces")
    def test_kind_selection(self):
        out = np.array([False], dtype=np.bool)
        m_type_test.write_array(out)

        self.assertEqual(out[0], True)

    def test_wrong_derived_type(self):
        out = np.array([-1], dtype=np.int32)

        with self.assertRaises(TypeError):
            m_type_test.is_circle_square(self._circle, out)

        with self.assertRaises(TypeError):
            m_type_test.is_circle_circle(self._square, out)

    def test_wrong_kind(self):
        out = np.array([-1], dtype=np.int64)

        with self.assertRaises(TypeError):
            m_type_test.write_array_int_1d(out)

    def test_wrong_type(self):
        out = np.array([-1], dtype=np.float32)

        with self.assertRaises(TypeError):
            m_type_test.write_array_int_1d(out)

    def test_wrong_dim(self):
        out = np.array([[-1]], dtype=np.int32)

        with self.assertRaises(TypeError):
            m_type_test.write_array_int_1d(out)

    def test_no_suitable_version(self):
        with self.assertRaises(TypeError):
            m_type_test.is_circle(1., 1.)

    def test_no_suitable_version_2(self):
        out = np.array([-1], dtype=np.complex128)

        with self.assertRaises(TypeError):
            m_type_test.write_array(out)

    def test_optional_scalar_real(self):
        out = np.array([-1], dtype=np.float32)
        opt_out = np.array(-1, dtype=np.float32)
        m_type_test.optional_scalar(out)
        self.assertEqual(out[0], 10)
        self.assertEqual(opt_out, -1)

        m_type_test.optional_scalar(out, opt_out)
        self.assertEqual(out[0], 10)
        self.assertEqual(opt_out, 20)

    def test_optional_scalar_int(self):
        out = np.array([-1], dtype=np.int32)
        opt_out = np.array(-1, dtype=np.int32)
        m_type_test.optional_scalar(out)
        self.assertEqual(out[0], 15)
        self.assertEqual(opt_out, -1)

        m_type_test.optional_scalar(out, opt_out)
        self.assertEqual(out[0], 15)
        self.assertEqual(opt_out, 25)

    def test_scalar_out_tolerance_real_32_64(self):
        out = np.array(-1, dtype=np.float32)
        with self.assertRaises(TypeError) as context:
          m_type_test.write_array_real64_0d(out)

    def test_scalar_out_tolerance_int_32_64(self):
        out = np.array(-1, dtype=np.int32)
        with self.assertRaises(TypeError) as context:
          m_type_test.write_array_int64_0d(out)

    def test_array_out_rigid_int_32_64(self):
        out = np.array([-1], dtype=np.int32)
        with self.assertRaises(TypeError) as context:
          m_type_test.write_array_int64_0d(out)

    def test_array_out_rigid_real_32_64(self):
        out = np.array([-1], dtype=np.float32)
        with self.assertRaises(TypeError) as context:
          m_type_test.write_array_double(out)

    def test_scalar_interface_int8(self):
        intput = np.array(-1, dtype=np.int8)
        out = m_type_test.in_scalar(intput)
        self.assertEqual(out, 108)

    def test_scalar_interface_int16(self):
        intput = np.array(-1, dtype=np.int16)
        out = m_type_test.in_scalar(intput)
        self.assertEqual(out, 116)

    def test_scalar_interface_int32(self):
        intput = np.array(-1, dtype=np.int32)
        out = m_type_test.in_scalar(intput)
        self.assertEqual(out, 132)

    def test_scalar_interface_int64(self):
        intput = np.array(-1, dtype=np.int64)
        out = m_type_test.in_scalar(intput)
        self.assertEqual(out, 164)

    def test_scalar_interface_float32(self):
        intput = np.array(-1, dtype=np.float32)
        out = m_type_test.in_scalar(intput)
        self.assertEqual(out, 232)

    def test_scalar_interface_float64(self):
        intput = np.array(-1, dtype=np.float64)
        out = m_type_test.in_scalar(intput)
        self.assertEqual(out, 264)

    def test_array_interface_int64(self):
        intput = np.array([-1], dtype=np.int64)
        out = m_type_test.in_scalar(intput)
        self.assertEqual(out, 364)

    def test_array_interface_float64(self):
        intput = np.array([-1], dtype=np.float64)
        out = m_type_test.in_scalar(intput)
        self.assertEqual(out, 464)

    def test_scalar_in_int8(self):
        intput = np.array(-1, dtype=np.int32)
        out = m_type_test.in_scalar_int8(intput)
        self.assertEqual(out, 108)

    def test_scalar_in_int16(self):
        intput = np.array(-1, dtype=np.int32)
        out = m_type_test.in_scalar_int16(intput)
        self.assertEqual(out, 116)

    def test_scalar_in_int32(self):
        intput = np.array(-1, dtype=np.int64)
        out = m_type_test.in_scalar_int32(intput)
        self.assertEqual(out, 132)

    def test_scalar_in_int64(self):
        intput = np.array(-1, dtype=np.int32)
        out = m_type_test.in_scalar_int64(intput)
        self.assertEqual(out, 164)

    def test_scalar_in_float32(self):
        intput = np.array(-1, dtype=np.float64)
        out = m_type_test.in_scalar_real32(intput)
        self.assertEqual(out, 232)

    def test_scalar_in_float64(self):
        intput = np.array(-1, dtype=np.float32)
        out = m_type_test.in_scalar_real64(intput)
        self.assertEqual(out, 264)

    def test_array_in_int64(self):
        intput = np.array([-1], dtype=np.int32)
        with self.assertRaises(TypeError) as context:
          out = m_type_test.in_array_int64(intput)

    def test_array_in_float64(self):
        intput = np.array([-1], dtype=np.float32)
        with self.assertRaises(TypeError) as context:
          out = m_type_test.in_array_real64(intput)

if __name__ == '__main__':

    unittest.main()
