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

import sys

pkg = False
if sys.argv[0].endswith('tests_pkg.py'):
    pkg = True
    import ExampleDerivedTypes_pkg
else:
    import ExampleDerivedTypes

class BaseTests(object):

    def setUp(self):
        pass

    def test_return_value_func(self):
        val_in = 100
        val_out = self.lib.library.return_value_func(val_in)
        self.assertEqual(val_in+10, val_out)

    def test_return_value_sub(self):
        val_in = 100
        val_out = self.lib.library.return_value_sub(val_in)
        self.assertEqual(val_in+10, val_out)

    def test_return_a_dt_func(self):
        dt = self.lib.library.return_a_dt_func()
        self.assert_(isinstance(dt, self.lib.datatypes.different_types))
        self.assertEqual(dt.alpha, 1) # logicals, so 1/0 instead of T/F
        self.assertEqual(dt.beta, 666)
        self.assertEqual(dt.delta, 666.666)

    def do_array_stuff(self, ndata, libfunc):
        x = np.arange(ndata, dtype=float)
        y = np.arange(ndata, dtype=float)
        br = np.zeros((ndata,), order='F')
        co = np.zeros((4, ndata), order='F')

        libfunc(n=ndata, x=x, y=y, br=br, co=co)

        for k in range(4):
            np.testing.assert_allclose(x*y + x, co[k,:])
        np.testing.assert_allclose(x/(y+1.0), br)

    def test_smallish_array(self):
        self.do_array_stuff(int(1e2), self.lib.library.do_array_stuff)

    def test_verybig_array(self):
        self.do_array_stuff(int(1e6), self.lib.library.do_array_stuff)

    def test_only_manipulate(self):
        n = int(1e5)
        x = np.arange(n, dtype=float)
        y = np.arange(n, dtype=float)
        br = np.zeros((n,), order='F')
        co = np.zeros((4, n), order='F')

        self.lib.library.do_array_stuff(n=n, x=x, y=y, br=br, co=co)
        self.lib.library.only_manipulate(n=n, array=co)

#        self.assert_(isinstance(br.dtype.type, np.float64))

        for k in range(4):
            np.testing.assert_allclose((x*y + x)**2, co[k,:])

    def test_set_derived_type(self):

        dt_beta = 99 # beta is an integer
        dt_delta = 199.0 # delta is double precision

        dt = self.lib.library.set_derived_type(dt_beta=dt_beta,
                                               dt_delta=dt_delta)
        self.assert_(isinstance(dt, self.lib.datatypes.different_types))
        self.assertEqual(dt.beta, dt_beta)
        self.assertEqual(dt.delta, dt_delta)

    def test_modify_derived_types(self):

        dt1 = self.lib.datatypes.different_types()
        dt1.beta = 10
        dt1.delta = 11.11
        dt2 = self.lib.datatypes.different_types()
        dt2.beta = 20
        dt2.delta = 22.22
        dt3 = self.lib.datatypes.different_types()
        dt3.beta = 30
        dt3.delta = 33.33

        self.lib.library.modify_derived_types(dt1, dt2, dt3)

        self.assertEqual(dt1.beta, 20)
        self.assertEqual(dt1.delta, 21.11)

        self.assertEqual(dt2.beta, 40)
        self.assertEqual(dt2.delta, 42.22)

        self.assertEqual(dt3.beta, 60)
        self.assertEqual(dt3.delta, 63.33)

    def test_modify_dertype_multiple_arrays(self):

        dt = self.lib.library.modify_dertype_fixed_shape_arrays()
        self.assert_(isinstance(dt, self.lib.datatypes.fixed_shape_arrays))

        self.assertEqual(dt.eta.dtype, np.int32)
        eta = np.ones((10,4), dtype=np.int)
        np.testing.assert_array_equal(dt.eta, eta*10)

        # FIXME: Fails because dt.theta is not float32 (single precision)
        # see test case: test_single_precesion_array
#        self.assertEqual(dt.theta.dtype, np.float32)
#        theta = np.ones((10,4), dtype=np.float32)*np.float32(2.0)
#        np.testing.assert_array_equal(dt.theta, theta)

        self.assertEqual(dt.iota.dtype, np.float64)
        iota = np.ones((10,4), dtype=np.float64)*np.float64(100.0)
        np.testing.assert_array_equal(dt.iota, iota)

    # this test will fail because theta is supposed to be single precision
    # expectedFailure only added to unittest module in Python 2.7
    # @unittest.expectedFailure
    # def test_single_precesion_array(self):

    #     dt = self.lib.library.modify_dertype_fixed_shape_arrays()
    #     self.assert_(isinstance(dt, self.lib.datatypes.Fixed_Shape_Arrays))

    #     # expected failure: dt.theta seems to be float64 (double precision)
    #     # while it should be single precision
    #     self.assertEqual(dt.theta.dtype, np.float32)
    #     theta = np.ones((10,4), dtype=np.float32)
    #     np.testing.assert_array_equal(dt.theta, theta*np.float32(2.0))

    def test_nested_dertype(self):
        ndt = self.lib.datatypes.nested()
        self.assert_(isinstance(ndt.mu.alpha, int)) # boolean/logical in F though)
        self.assert_(isinstance(ndt.mu.beta, int))
        self.assert_(isinstance(ndt.mu.delta, float))
        self.assert_(isinstance(ndt.nu, self.lib.datatypes.fixed_shape_arrays))

    def test_return_dertype_pointer_arrays(self):
        m, n = 9, 5
        dt = self.lib.library.return_dertype_pointer_arrays(m, n)
        self.assert_(isinstance(dt, self.lib.datatypes.pointer_arrays))
        expected = np.ones((m, n), order='F', dtype=np.float64) * 100.0
        expected[m-3, n-2] = -10.0
        np.testing.assert_allclose(dt.chi, expected)

    def test_return_dertype_alloc_arrays(self):
        m, n = 9, 5
        dt = self.lib.library.return_dertype_alloc_arrays(m, n)
        dt.chi_shape = np.array([m, n], dtype=np.int32)
        self.assert_(isinstance(dt, self.lib.datatypes_allocatable.alloc_arrays))
        expected = np.ones((m, n), order='F', dtype=np.float64) * 10.0
        expected[m-3, n-2] = -1.0
        np.testing.assert_allclose(dt.chi, expected)

    def test_modify_dertype_pointer_arrays(self):
        # the array can only be created from Fortran and this will fail
        # dt = self.lib.datatypes.Pointer_Arrays()
        # m, n = 9, 5
        # dt.chi = np.mgrid[0:m,0:n][0]
        # dt.chi_shape = np.array([m,n], dtype=np.int32)

        # create an array in Fortran
        m, n = 9, 5
        dt = self.lib.library.return_dertype_pointer_arrays(m, n)
        # the shape of the 2D array is also part of the derived type
        # so we do not have to pass the shape of the array again when modifying
        # the those arrays in other subroutines
        dt.chi_shape = np.array([m, n], dtype=np.int32)
        # we can also place a new array in here as long as we keep the same
        # data type
        dt.chi = np.ndarray((m,n), dtype=np.float64)
        dt.chi[:,:] = 9.0
        dt.chi[0,0] = 50.0
        self.lib.library.modify_dertype_pointer_arrays(dt)

        expected = np.ones((m, n), order='F', dtype=np.float64) * 81.0
        expected[0, 0] = 2500.0
        expected[m-3, n-2] = -9.0
        np.testing.assert_allclose(dt.chi, expected)

    def test_modify_dertype_alloc_arrays(self):

#        # the array can only be created from Fortran and this will fail
#        dt = self.lib.datatypes_allocatable.Alloc_Arrays()
#        dt.chi = np.mgrid[0:9,0:6][0]
#        dt.chi_shape = np.ndarray([9,6], dtype=np.int32)

        # create an array in Fortran
        m, n = 9, 5
        dt = self.lib.library.return_dertype_alloc_arrays(m, n)
        # the shape of the 2D array is also part of the derived type
        # so we do not have to pass the shape of the array again when modifying
        # the those arrays in other subroutines
        dt.chi_shape = np.array([m, n], dtype=np.int32)
        # we can also place a new array in here as long as we keep the same
        # data type
        dt.chi = np.ndarray((m,n), dtype=np.float64)
        dt.chi[:,:] = 9.0
        dt.chi[0,0] = 50.0
        self.lib.library.modify_dertype_alloc_arrays(dt)

        expected = np.ones((m, n), order='F', dtype=np.float64) * 81.0
        expected[0, 0] = 2500.0
        expected[m-3, n-2] = -9.0
        np.testing.assert_allclose(dt.chi, expected)

#    def test_array_of_dertype(self):
#        dt = self.lib.datatypes.Array_Nested

    def test_alloc_arrays(self):
        m, n = 10, 4
        dt = self.lib.datatypes_allocatable.alloc_arrays()
        self.lib.datatypes_allocatable.init_alloc_arrays(dt, m, n)
        self.assertEqual(dt.chi.shape, (m,n))

        dt.chi = np.arange(m*n).reshape(m, n)

        def assign_wrong_size():
            dt.chi = np.arange(m*(n-1)).reshape(m, n-1)
        self.assertRaises(ValueError, assign_wrong_size)

    def test_nested_alloc_arrays(self):
        n = 10
        dt = self.lib.datatypes.array_nested()
        self.lib.datatypes.init_array_nested(dt, n)

        self.assertEqual(len(dt.xi), n)
        self.assertEqual(len(dt.omicron), n)
        self.assertEqual(len(dt.pi), n)



if pkg:
    class LibTestsPkg(unittest.TestCase, BaseTests):
        lib = ExampleDerivedTypes_pkg
else:
    class LibTests(unittest.TestCase, BaseTests):
        lib = ExampleDerivedTypes


if __name__ == '__main__':

    unittest.main()
