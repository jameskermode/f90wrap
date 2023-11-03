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
@author: Caoxiang Zhu
"""

from __future__ import print_function

import unittest

import numpy as np

import keywordr_rename as lib


def test():
    a = 3
    # normal type
    assert lib.global_.abc == 0
    # rename keyword in normal types
    assert lib.global_.lambda_ == 1
    # rename keyword in derived types
    y = lib.global_.class2()
    assert y.x == 456
    # rename keyword in array
    assert len(lib.global_.with_) == 9
    # rename keyword in subroutine
    lib.global_.is_(a)
    assert lib.global_.abc == a
    # rename keyword in function
    assert lib.in_(a) == a + 1


if __name__ == "__main__":
    test()
