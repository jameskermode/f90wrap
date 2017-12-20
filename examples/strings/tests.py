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


import ExampleStrings
import ExampleStrings_pkg


def generate_string(n):
    s = ''.join([chr(k) for k in range(34,n+34)])
    return s.encode('latin-1')


class BaseTests(object):

    text = b'-_-::this is a string with ASCII, / and 123...::-_-'
    n = len(text)

    def test_func_generate_string(self):
        outstring = self.lib.func_generate_string(self.n)
        self.assertEqual(outstring[:self.n], generate_string(self.n))

    def test_func_return_string(self):
        outstring = self.lib.func_return_string()
        self.assertEqual(outstring[:self.n], self.text)

    def test_generate_string(self):
        outstring = self.lib.generate_string(self.n)
        self.assertEqual(outstring[:self.n], generate_string(self.n))

    def test_return_string(self):
        outstring = self.lib.return_string()
        self.assertEqual(outstring[:self.n], self.text)

    def test_set_global_string(self):
        self.lib.set_global_string(self.n, self.text)
        if hasattr(self.lib, 'global_string'):
            # module mode, has global_string property
            self.assertEqual(self.lib.global_string.strip(), self.text)
        else:
            # package mode, has get_global_string accessor
            self.assertEqual(self.lib.get_global_string().strip(), self.text)


    # string on an inout variable does not work
    # expectedFailure only added to unittest module in Python 2.7
#    @unittest.expectedFailure
#    def test_inout_string(self):
#        stringio = ' '*self.n
#        stringio = self.lib.inout_string(self.n, stringio)
#        self.assertEqual(stringio, 'Z'*self.n)


class LibTests(unittest.TestCase, BaseTests):
    lib = ExampleStrings.string_io


class LibTestsPkg(unittest.TestCase, BaseTests):
    lib = ExampleStrings_pkg.string_io


if __name__ == '__main__':
    unittest.main()
