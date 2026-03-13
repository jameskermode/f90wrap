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
from __future__ import print_function

import unittest

import numpy as np
from pywrapper import list_strings

initial_array = np.array(["a", "ab", "abc"])

class BaseTests(unittest.TestCase):

    def test_get_attribute(self):
        a = list_strings.mytype()
        assert(np.all(a.strings == initial_array))

    def test_set_attribute_with_list(self):
        a = list_strings.mytype()
        new = ["x1", "yz", "vvv"]
        a.strings = new
        assert(np.all(a.strings == np.array(new)))

    def test_set_attribute_with_numpy(self):
        a = list_strings.mytype()
        new = ["x1", "yz", "vvv"]
        a.strings = new
        assert(np.all(a.strings == new))

    def test_set_get_attribute(self):
        a = list_strings.mytype()
        a.strings = a.strings
        assert(np.all(a.strings == initial_array))

    def test_get_function(self):
        a = list_strings.mytype()
        assert(np.all(list_strings.mytype_get_strings(a) == initial_array))

    def test_set_function_with_list(self):
        a = list_strings.mytype()
        new = ["x", "y", "z"]
        # Pywrapper expects a numpy array here
        with self.assertRaises(TypeError):
            list_strings.mytype_set_strings(a, new)

    def test_set_function_with_numpy(self):
        a = list_strings.mytype()
        new = np.array(["x1", "yz", "vvv"])
        list_strings.mytype_set_strings(a, new)
        assert(np.all(a.strings == new))
        assert(np.all(list_strings.mytype_get_strings(a) == new))

    def test_set_function_implicit_with_numpy(self):
        a = list_strings.mytype()
        new = np.array(["x1", "yz", "vvv"])
        list_strings.mytype_set_strings_implicit(a, new)
        assert(np.all(a.strings == new))
        assert(np.all(list_strings.mytype_get_strings(a) == new))

    def test_set_function_single(self):
        a = list_strings.mytype()
        new = np.array(["x1", "yz", "vvv"])
        list_strings.mytype_set_string(a, new[0], 1)
        list_strings.mytype_set_string(a, new[1], 2)
        list_strings.mytype_set_string(a, new[2], 3)
        assert(np.all(a.strings == new))
        assert(np.all(list_strings.mytype_get_strings(a) == new))

if __name__ == '__main__':
    unittest.main()
