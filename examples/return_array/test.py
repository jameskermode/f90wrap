import unittest
import numpy as np

from pywrapper import m_test

class TestReturnArray(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestReturnArray, self).__init__(*args, **kwargs)
        self._shape = (2,)
        self._array = m_test.t_array_wrapper()
        m_test.array_init(self._array, *self._shape)


    def test_init(self):
        array = m_test.t_array_wrapper()
        m_test.array_init(array, 2)

    def test_free(self):
        array = m_test.t_array_wrapper()
        m_test.array_init(array, 2)
        m_test.array_free(array)

    def test_return_scalar(self):
        out_scalar = m_test.return_scalar(self._array)

        assert(isinstance(out_scalar, float))
        assert(out_scalar == 1)

    def test_return_hard_coded_1d(self):
        out_array = m_test.return_hard_coded_1d()

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == (10,))
        assert((out_array == 2).all)

    def test_return_hard_coded_2d(self):
        out_array = m_test.return_hard_coded_2d()

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == (5,6))
        assert((out_array == 3).all)

    def test_return_array_member(self):
        out_array = m_test.return_array_member(self._array)

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == self._shape)
        assert((out_array == 1).all)

    def test_return_array_member_2d(self):
        shape = (3,4)
        array_2d = m_test.t_array_2d_wrapper()
        m_test.array_2d_init(array_2d, *shape)

        out_array = m_test.return_array_member_2d(array_2d)

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == shape)
        assert((out_array == 2).all)

    def test_return_array_member_wrapper(self):
        shape = (3,)
        array_wrapper = m_test.t_array_double_wrapper()
        m_test.array_wrapper_init(array_wrapper, *shape)

        out_array = m_test.return_array_member_wrapper(array_wrapper)

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == shape)
        assert((out_array == 2).all)

    def test_return_array_input(self):
        shape = (4,)
        out_array = m_test.return_array_input(*shape)

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == shape)
        assert((out_array == 1).all)

    def test_return_array_input_2d(self):
        shape = (5,4)
        out_array = m_test.return_array_input_2d(*shape)

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == shape)
        assert((out_array == 2).all)

    def test_return_array_size(self):
        shape = (2,)
        in_array = np.zeros(shape, dtype=np.float32, order="F")
        out_array = m_test.return_array_size(in_array)

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == shape)
        assert((out_array == 1).all)

    def test_return_array_size_2d_in(self):
        shape = (2, 3)
        in_array = np.zeros(shape, dtype=np.float32, order="F")
        out_array = m_test.return_array_size_2d_in(in_array)

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == (shape[1],))
        assert((out_array == 1).all)

    def test_return_array_size_2d_out(self):
        shape_1 = (2, 3)
        shape_2 = (5, 4)
        in_array_1 = np.zeros(shape_1, dtype=np.float32, order="F")
        in_array_2 = np.zeros(shape_2, dtype=np.float32, order="F")
        out_array = m_test.return_array_size_2d_out(in_array_1, in_array_2)

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == (shape_1[0], shape_2[1]))
        assert((out_array == 2).all)

    def test_return_derived_type_value(self):
        out_value = 1
        value_type = m_test.t_value()
        value_type.value = out_value
        shape = (2, 3)
        size_2d = m_test.t_size_2d()
        size_2d.x = shape[0]
        size_2d.y = shape[1]

        out_array = m_test.return_derived_type_value(value_type, size_2d)

        assert(isinstance(out_array, np.ndarray))
        assert(out_array.shape == shape)
        assert((out_array == out_value).all)

if __name__ == '__main__':

    unittest.main()
