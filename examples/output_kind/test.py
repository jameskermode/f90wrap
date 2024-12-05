import unittest
import numpy as np

from pywrapper import m_out_test

class TestTypeCheck(unittest.TestCase):
  def test_out_scalar_int1(self):
    out = m_out_test.out_scalar_int1()
    self.assertEqual(1,out)
    self.assertIsInstance(out, int)

  def test_out_scalar_int2(self):
    out = m_out_test.out_scalar_int2()
    self.assertEqual(2,out)
    self.assertIsInstance(out, int)

  def test_out_scalar_int4(self):
    out = m_out_test.out_scalar_int4()
    self.assertEqual(4,out)
    self.assertIsInstance(out, int)

  def test_out_scalar_int8(self):
    out = m_out_test.out_scalar_int8()
    self.assertEqual(8,out)
    self.assertIsInstance(out, int)

  def test_out_scalar_real4(self):
    out = m_out_test.out_scalar_real4()
    self.assertEqual(4.,out)
    self.assertIsInstance(out, float)

  def test_out_scalar_real8(self):
    out = m_out_test.out_scalar_real8()
    self.assertEqual(8.,out)
    self.assertIsInstance(out, float)

  def test_out_array_int4(self):
    out = m_out_test.out_array_int4()
    self.assertEqual(4,out[0])
    self.assertIsInstance(out, np.ndarray)
    self.assertIsInstance(out[0], np.intc)
    self.assertIsInstance(out[0], np.int32)

  def test_out_array_int8(self):
    out = m_out_test.out_array_int8()
    self.assertEqual(8,out[0])
    self.assertIsInstance(out, np.ndarray)
    self.assertIsInstance(out[0], np.int_)
    self.assertIsInstance(out[0], np.int64)
    self.assertNotIsInstance(out[0], np.longlong)

  def test_out_array_real4(self):
    out = m_out_test.out_array_real4()
    self.assertEqual(4.,out[0])
    self.assertIsInstance(out, np.ndarray)
    self.assertIsInstance(out[0], np.single)
    self.assertIsInstance(out[0], np.float32)

  def test_out_array_real8(self):
    out = m_out_test.out_array_real8()
    self.assertEqual(8.,out[0])
    self.assertIsInstance(out, np.ndarray)
    self.assertIsInstance(out[0], np.double)
    self.assertIsInstance(out[0], np.float64)

if __name__ == '__main__':

    unittest.main()
