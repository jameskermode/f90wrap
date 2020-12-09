#  f90wrap: F90 to Python interface generator with derived type support
#
#  Copyright James Kermode 2011-2020
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
import test

assert hasattr(test, 'routine_member_procedures')
assert not hasattr(test, 'member_procedure')
assert not hasattr(test, 'member_function')

out1, out2 = test.routine_member_procedures(1, 2)
assert out1 == 7  # 5*1+2
assert out2 == 23 # 3*out1+2

assert hasattr(test, 'routine_member_procedures2')
assert not hasattr(test, 'member_procedure2')
assert not hasattr(test, 'member_function2')

out12, out22 = test.routine_member_procedures2(1, 2)
assert out12 == 28 # (out2-10)*2+2
assert out22 == 84 # out12*3
