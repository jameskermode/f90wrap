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
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:19:03 2015

@author: David Verelst
"""

from __future__ import print_function

import unittest

import numpy as np

import testmodule as lib


class TestExample(unittest.TestCase):

    def setUp(self):
        pass

    def do_array_stuff(self, n=3, m=4):

        print("n,m=",n,m)
        x = lib.my_module.mytype()
        lib.my_module.allocit(x,n,m)

        sum = 0.0
        for k in range(n):
              for j in range(m):
                  sum += x.y[k,j]
        print("sum = %20.10f  .... x.y[%d,%d] = %20.10f \n" % ( sum, n,m,x.y[n-1,m-1] ) )

    def test_basic(self):
        self.do_array_stuff(1,2)

    def test_normal_array(self):
        self.do_array_stuff(10,20)

    def test_verybig_array(self):
        self.do_array_stuff(50,99)

if __name__ == '__main__':

    unittest.main()
