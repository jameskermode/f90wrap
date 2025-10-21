import unittest
import numpy as np

from pywrapper import m_array

class TestsArray(unittest.TestCase):
    def test_init(self):
        array = m_array.Array()
        array.init(4)
        self.assertEqual(array.array_size, 4)
        self.assertEqual(array.buffer.shape, (4,))
        self.assertTrue(np.all(array.buffer == 0))

    def test_init_optional(self):
        array = m_array.Array()
        optional = m_array.Array()
        optional.init(4)
        array.init_optional(4, optional_arg=optional)
        self.assertEqual(array.array_size, 4)
        self.assertEqual(array.buffer.shape, (4,))
        self.assertTrue(np.all(array.buffer == 0))

    def test_init_optional_none(self):
        array = m_array.Array()
        array.init_optional(4)
        self.assertEqual(array.array_size, 4)
        self.assertEqual(array.buffer.shape, (4,))
        self.assertTrue(np.all(array.buffer == 0))

if __name__ == '__main__':
    unittest.main()
