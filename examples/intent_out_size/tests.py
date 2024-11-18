import unittest
import numpy as np

from pywrapper import m_intent_out

class TestIntentOut(unittest.TestCase):

  def test_intent_out_size(self):

    a1 = np.array([[1,2], [3,4]], dtype=np.float32, order='F')
    a2 = np.array([[2,4], [6,8]], dtype=np.float32, order='F')
    output = np.zeros((2,2), dtype=np.float32, order='F')
    n1 = 2
    n2 = 2

    m_intent_out.interpolation(n1,n2,a1,a2,output)

    ref_out = np.array([[1.5,3.], [4.5,6.]], dtype=np.float32, order='F')

    np.testing.assert_array_equal(output, ref_out)

if __name__ == '__main__':

  unittest.main()
