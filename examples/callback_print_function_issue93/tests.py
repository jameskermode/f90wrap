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
import sys

import numpy as np

import CBF

class TestExample(unittest.TestCase):

    def setUp(self):
        pass

    def test_basic(self):
        """Test f2py callbacks through cross-module Fortran calls.

        Note: Do NOT call write_message directly before calling through caller
        module. The f2py callback wrapper has a bug where it stores the callback
        context in thread-local storage but doesn't restore it after the call
        returns. This leaves a dangling pointer that causes segfaults when the
        callback is invoked through a different path (e.g., via caller module).

        See: https://github.com/jameskermode/f90wrap/issues/93
        """
        print(CBF._CBF.cback.write_message.__doc__)
        def f(msg):
            print("Yo! " + msg)
        CBF._CBF.pyfunc_print = f
        # Call through caller module - this works correctly
        CBF.caller.test_write_msg()
        CBF.caller.test_write_msg()
        CBF.caller.test_write_msg_2()

if __name__ == '__main__':
    unittest.main()
