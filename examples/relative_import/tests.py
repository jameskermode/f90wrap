import unittest
import numpy as np

from top_module.pywrapper import m_fortran_module
from top_module.pywrapper import m_base_type
from top_module.pywrapper import m_inheritance
from top_module.pywrapper import m_composition

class TestImport(unittest.TestCase):

  def test_subroutine_type(self):
    obj = m_base_type.t_base_type()
    self.assertIsInstance(obj, m_base_type.t_base_type)
    self.assertEqual(obj.real_number, 0.0)
    m_fortran_module.a_subroutine(obj)
    self.assertEqual(obj.real_number, 1.0)

  def test_subroutine_child_type(self):
    obj = m_inheritance.t_inheritance()
    self.assertIsInstance(obj, m_inheritance.t_inheritance)
    self.assertEqual(obj.real_number, 0.0)
    self.assertEqual(obj.integer_number, 0)
    m_fortran_module.b_subroutine(obj)
    self.assertEqual(obj.real_number, 1.0)
    self.assertEqual(obj.integer_number, 2)

  def test_subroutine_with_member(self):
    obj = m_composition.t_composition()
    self.assertIsInstance(obj, m_composition.t_composition)
    self.assertEqual(obj.member.real_number, 0.0)
    m_fortran_module.c_subroutine(obj)
    self.assertEqual(obj.member.real_number, 1.0)

if __name__ == '__main__':

    unittest.main()
