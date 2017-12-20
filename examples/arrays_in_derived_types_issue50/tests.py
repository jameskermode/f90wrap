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
from issue50 import module_test as tp
from numpy import zeros, ones, float32, abs, max

a = tp.real_array()
print("This is the freshly allocated array : " + str(a.item))
a.item = ones(6, dtype=float32) 
print("This is sent to fortran : " + str(a.item))
tp.testf(a)
print("This is received by python : " + str(a.item))

assert max(abs(a.item - [1.0, 1.0, 1.0, 4.0, 1.0, 1.0])) < 1e-6
