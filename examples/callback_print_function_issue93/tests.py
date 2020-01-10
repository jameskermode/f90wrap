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
Created on Tue Jan 25 2018

@author: Gaetan Kenway
"""

from __future__ import print_function

import unittest

import numpy as np

import CBF

class TestExample(unittest.TestCase):

    def setUp(self):

        pass

    def test_basic(self):
        print(CBF._CBF.cback.write_message.__doc__)
        def f(msg): 
            print("Yo! " + msg)
        CBF._CBF.pyfunc_print = f
        # We need to prime the callback with a call "under the hood", not sure why.
        CBF._CBF.cback.write_message('blah')
        # Subsequently other calls to higher level functions work.
        CBF.caller.test_write_msg()
        CBF.caller.test_write_msg()
        CBF.caller.test_write_msg_2()
        # TODO?
        # CBF.caller.test_return_msg()
        # CBF.caller.test_return_msg()
        # CBF.caller.test_return_msg_2()

if __name__ == '__main__':
    unittest.main()
