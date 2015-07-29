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
        self.do_array_stuff(1e3)

    def test_verybig_array(self):
        self.do_array_stuff(1e6)

    def test_square(self):
        n = 1e5
        x = np.arange(n)
        y = np.arange(n)
        br = np.zeros((n,), order='F')
        co = np.zeros((4, n), order='F')

        lib.library.do_array_stuff(n=n, x=x, y=y, br=br, co=co)
        lib.library.only_manipulate(n=n, array=co)
        for k in range(4):
            np.testing.assert_allclose((x*y + x)**2, co[k,:])


if __name__ == '__main__':

    unittest.main()
