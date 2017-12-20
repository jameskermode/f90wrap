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

import ExampleArray as lib


class TestExample(unittest.TestCase):

    def setUp(self):
        pass

    def do_array_stuff(self, ndata):
        x = np.arange(ndata)
        y = np.arange(ndata)
        br = np.zeros((ndata,), order='F')
        co = np.zeros((4, ndata), order='F')

        lib.library.do_array_stuff(n=ndata, x=x, y=y, br=br, co=co)

        for k in range(4):
            np.testing.assert_allclose(x*y + x, co[k,:])
        np.testing.assert_allclose(x/(y+1.0), br)

    def test_basic(self):
        self.do_array_stuff(1000)

    def test_verybig_array(self):
        self.do_array_stuff(1000000)

    def test_square(self):
        n = 100000
        x = np.arange(n, dtype=float)
        y = np.arange(n, dtype=float)
        br = np.zeros((n,), order='F')
        co = np.zeros((4, n), order='F')

        lib.library.do_array_stuff(n=n, x=x, y=y, br=br, co=co)
        lib.library.only_manipulate(n=n, array=co)
        for k in range(4):
            np.testing.assert_allclose((x*y + x)**2, co[k,:])

    def test_return_array(self):
        m, n = 10, 4
        arr = np.ndarray((m,n), order='F', dtype=np.int32)
        lib.library.return_array(m, n, arr)
        ii, jj = np.mgrid[0:m,0:n]
        ii += 1
        jj += 1
        np.testing.assert_equal(ii*jj + jj, arr)


if __name__ == '__main__':

    unittest.main()
