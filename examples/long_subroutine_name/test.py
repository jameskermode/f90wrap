import unittest

from pywrapper import m_long_subroutine_name

class TestLongSubroutineName(unittest.TestCase):

    def test_long_subroutine_name(self):

        m_long_subroutine_name.m_long_subroutine_name_integer = 42

        typ = m_long_subroutine_name.m_long_subroutine_name_type()
        typ.m_long_subroutine_name_type_name_integer = 42
        typ.m_long_subroutine_name_type_name_integer_array = 42

        typ2 = m_long_subroutine_name.m_long_subroutine_name_type_2()
        typ2.m_long_subroutine_name_type_2_type_array[0].m_long_subroutine_name_type_integer = 42
        typ2.m_long_subroutine_name_type_2_type_array[0].m_long_subroutine_name_type_integer_array = 42
        
        m_long_subroutine_name.m_long_subroutine_name_subroutine()

if __name__ == '__main__':

    unittest.main()
