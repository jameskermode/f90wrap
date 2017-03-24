
from __future__ import print_function

import unittest

import numpy as np

import Example


class TestExample(unittest.TestCase):

    def setUp(self):
        pass

    def test_auto_diff(self):
        d1 = Example.Dual_Num_Auto_Diff.DUAL_NUM()
        d2 = Example.Dual_Num_Auto_Diff.DUAL_NUM()
        d3 = Example.Mcyldnad.cyldnad(d1, d2)

if __name__ == '__main__':
    unittest.main()
