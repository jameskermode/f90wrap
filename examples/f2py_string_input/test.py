import unittest
import numpy as np

import _pywrapper

class TestString(unittest.TestCase):

    @unittest.skip("Suspected f2py bug https://github.com/numpy/numpy/issues/24706")
    def test_string_in_array(self):
        in_array = np.array(['one', 'two'], dtype='c')
        output = _pywrapper.string_in_array(in_array)
        self.assertEqual(output, 0)

if __name__ == '__main__':

    unittest.main()
