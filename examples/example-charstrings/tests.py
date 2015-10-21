# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:19:03 2015

@author: David Verelst
"""

from __future__ import print_function

import unittest

import numpy as np

import ExampleCharStrings
import ExampleCharStrings_pkg


class BaseTests(object):

    text = '-_-::this is a string with ASCII, / and 123...::-_-'
    intarray = np.array([ord(k) for k in list(text)], dtype=np.int32)
    chararray = np.array([k for k in list(text)], dtype='c')
    n = len(text)

    def test_roundtrip(self):
        intarray_out = np.ndarray((len(self.text),), dtype=np.int32)
        self.lib.roundtrip(len(self.text), self.intarray, intarray_out)
        np.testing.assert_equal(intarray_out, self.intarray)

    # test will fail because passing character arrays like this does not work
    # expectedFailure only added to unittest module in Python 2.7
#    @unittest.expectedFailure
#    def test_chararray(self):
#        self.lib.testing_chararray(self.n, self.chararray)

    def test_intarray(self):
        self.lib.testing_intarray(self.n, self.intarray)

    # test will fail because passing character arrays like this does not work
    # expectedFailure only added to unittest module in Python 2.7
#    @unittest.expectedFailure
#    def test_chararray_output(self):
#        chararray_out = self.lib.intarray2chararray(self.n, self.intarray)

    # test will fail because passing character arrays like this does not work
    # expectedFailure only added to unittest module in Python 2.7
#    @unittest.expectedFailure
#    def test_chararray_roundtrip(self):
#        chararray_out = np.ndarray((self.n,), dtype='c')
#        self.lib.chararray_roundtrip(self.n, self.chararray, chararray_out)

    def test_fill_global_string(self):
        self.lib.fill_global_string(self.n, self.intarray)


class LibTests(unittest.TestCase, BaseTests):
    lib = ExampleCharStrings.stringutils


class LibTestsPkg(unittest.TestCase, BaseTests):
    lib = ExampleCharStrings_pkg.stringutils


if __name__ == '__main__':
    unittest.main()
