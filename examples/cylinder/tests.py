#  f90wrap: F90 to Python interface generator with derived type support
#
#  Copyright James Kermode 2011-2018
#
#  This file is part of f90wrap
#  For the latest version see github.com/jameskermode/f90wrap
#
#  f90wrap is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  f90wrap is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with f90wrap. If not, see <http://www.gnu.org/licenses/>.
# 
#  If you would like to license the source code under different terms,
#  please contact James Kermode, james.kermode@gmail.com

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
