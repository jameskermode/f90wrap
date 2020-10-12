"""
Created on Mon Oct 12, 2020

@author: Bernardo Pacini
"""
import unittest
import numpy as np

from recursive_type_array import Mod_Recursive_Type_Array as recursive_type_array

class Test_recursive_type_array(unittest.TestCase):
    def setUp(self):
        self.N_node = 3

        self.root = recursive_type_array.t_node()

        recursive_type_array.allocate_node(self.root,self.N_node)

    def tearDown(self):
        recursive_type_array.deallocate_node(self.root)

    def test_allocate(self):
        self.assertEqual(
            len(self.root.node),
            self.N_node
        )

    def test_deallocate(self):
        recursive_type_array.deallocate_node(self.root)
        self.assertEqual(
            len(self.root.node),
            0
        )

if __name__ == "__main__":
    unittest.main()
