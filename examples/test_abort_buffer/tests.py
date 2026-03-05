import unittest

# The import order is important here
# If pywrapper_main_A is imported first everything is good
# If pywrapper_main_B is imported first a segmentation fault occurs
from pywrapper_main_B import m_test_b
from pywrapper_main_A import m_test_a

class TestCallAbort(unittest.TestCase):

    def test_call_abort_a(self):
        with self.assertRaises(RuntimeError):
            m_test_a.calling_abort_a()

    def test_call_abort_b(self):
        with self.assertRaises(RuntimeError):
            m_test_b.calling_abort_b()

if __name__ == '__main__':

    unittest.main()
