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
