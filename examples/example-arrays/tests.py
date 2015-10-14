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
        x = np.arange(n)
        y = np.arange(n)
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
