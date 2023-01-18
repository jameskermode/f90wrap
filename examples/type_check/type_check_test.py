import unittest
import numpy as np

from pywrapper import m_circle

class TestTypeCheck(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTypeCheck, self).__init__(*args, **kwargs)
        self._circle = m_circle.t_circle()
        self._square = m_circle.t_square()

    def test_derived_type_selection(self):
        out_circle = np.array([-1], dtype=np.int32)
        out_square = np.array([-1], dtype=np.int32)

        m_circle.is_circle(self._circle, out_circle)
        m_circle.is_circle(self._square, out_square)

        assert out_circle[0]==1
        assert out_square[0]==0

    def test_shape_selection_1d(self):
        out = np.array([-1], dtype=np.int32)
        m_circle.write_array(out)

        assert out[0]==1

    def test_shape_selection_2d(self):
        out = np.array([[-1]], dtype=np.int32)
        m_circle.write_array(out)

        assert out[0]==2

    def test_type_selection(self):
        out = np.array([-1], dtype=np.float32)
        m_circle.write_array(out)

        assert out[0]==3

    def test_kind_selection(self):
        out = np.array([-1], dtype=np.float64)
        m_circle.write_array(out)

        assert out[0]==4

    def test_wrong_derived_type(self):
        out = np.array([-1], dtype=np.int32)

        with self.assertRaises(TypeError):
            m_circle._is_circle_square(self._circle, out)

        with self.assertRaises(TypeError):
            m_circle._is_circle_circle(self._square, out)

    def test_wrong_kind(self):
        out = np.array([-1], dtype=np.int64)

        with self.assertRaises(TypeError):
            m_circle._write_array_int_1d(out)

    def test_wrong_type(self):
        out = np.array([-1], dtype=np.float32)

        with self.assertRaises(TypeError):
            m_circle._write_array_int_1d(out)

    def test_wrong_dim(self):
        out = np.array([[-1]], dtype=np.int32)

        with self.assertRaises(TypeError):
            m_circle._write_array_int_1d(out)

    def test_no_suitable_version(self):
        with self.assertRaises(TypeError):
            m_circle.is_circle(1., 1.)

    def test_no_suitable_version_2(self):
        out = np.array([-1], dtype=np.complex)

        with self.assertRaises(TypeError):
            m_circle.write_array(out)


if __name__ == '__main__':

    unittest.main()
