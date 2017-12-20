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
import test_python
import numpy

a = numpy.ones(test_python.test_module.m, dtype=numpy.float32)
b = test_python.test_module.Test_Type2_Xn_Array()
c = test_python.test_module.Test_Type2_Xn_Array()
d = test_python.test_module.Test_Type2_Xm_Array()
e = test_python.test_module.Test_Type2_X5_Array()
f = numpy.ones(1)

test_python.test_module.test_routine4(a, b, c, d, e, f)

print(a)
print(list(b.items[i].y for i in range(len(b.items))))
print(list(c.items[i].y for i in range(len(c.items))))
print(list(d.items[i].y for i in range(len(d.items))))
print(list(e.items[i].y for i in range(len(e.items))))
print(f)


assert(all(a == numpy.array([42, 1, 1, 1, 1], dtype=numpy.float32)))
assert(b.items[1].y[1] == 42)
assert(c.items[2].y[2] == 42)
assert(d.items[3].y[3] == 42)
assert(e.items[4].y[4] == 42)
assert(f == 2)
