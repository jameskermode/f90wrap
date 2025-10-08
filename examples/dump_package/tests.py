import unittest
import numpy as np

from pywrapper_array_type import m_array_type
from pywrapper_main import m_array_init

class TestsArray(unittest.TestCase):
    def test_init(self):
        array = m_array_type.Array()
        m_array_init.array_init(array, 4)
        self.assertEqual(array.array_size, 4)
        self.assertEqual(array.buffer.shape, (4,))
        self.assertTrue(np.all(array.buffer == 0))

if __name__ == '__main__':
    unittest.main()
