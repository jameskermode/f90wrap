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
import numpy as np

from mockdt import (define_a_type,
                    leveltwomod,
                    use_a_type,
                    top_level)

a = define_a_type.atype() # calls initialise()

a.rl = 3.0 # calls set()
assert(a.rl == 3.0)

a.vec[:] = 0. # calls get() then sets array data in place
assert(np.all(a.vec == 0.0))
a.vec = 1.
assert(np.all(a.vec == 1.0))

a.dtype.rl = 1.0 # calls set()

my_l2 = leveltwomod.leveltwo(4.0) # calls initialise()
a.dtype = my_l2 # calls set()
assert(a.dtype.rl == my_l2.rl)

# access the module-level variables in use_a_type
use_a_type.p.rl = 1.0
use_a_type.p.bool = True
use_a_type.p.integ = 10

# call a routine in use_a_type
result = use_a_type.do_stuff(8)
assert(result == 1073741824.0)

# now we can access the array of derived types in use_a_type
assert(len(use_a_type.p_array) == 3)
assert(use_a_type.p.rl == use_a_type.p_array[0].rl)
assert(use_a_type.p.rl == use_a_type.p_array[1].rl)
assert(use_a_type.p.rl == use_a_type.p_array[2].rl)

use_a_type.vector[:] = 1.0
assert(np.all(use_a_type.vector == 1.0))

input = 3.0
output = top_level(input)
assert(output == 85.0*input)

# test access to array of derived types p_array
assert(len(use_a_type.p_array) == 3)
for i in range(len(use_a_type.p_array)):
    assert(use_a_type.p_array[i].bool == use_a_type.p.bool)
    assert(use_a_type.p_array[i].integ == use_a_type.p.integ)
    assert(use_a_type.p_array[i].rl == use_a_type.p.rl)
    assert(np.all(use_a_type.p_array[i].vec == use_a_type.p.vec))

# test function with derived type return type
a = define_a_type.return_a_type_func()
assert(a.bool == 1 or a.bool == -1) # ifort uses -1 for logical true
assert(a.integ == 42)
    
# test subroutine with intent(out) derived type argument
a = define_a_type.return_a_type_sub()
assert(a.bool == 1 or a.bool == -1) # ifort uses -1 for logical true
assert(a.integ == 42)
