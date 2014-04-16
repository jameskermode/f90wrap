from define_a_type import *
from leveltwomod import *
from use_a_type import *

a = Atype() # calls initialise()

a.rl = 3.0 # calls set()
print 'a.rl =', a.rl # calls get()

a.vec[:] = 0. # calls get() then sets array data in place
print 'a.vec =', a.vec # calls get()
a.vec = 1. # calls set()
print 'a.vec =', a.vec # calls get()

a.dtype.rl = 1.0 # calls set()
a.dtype.print_() # calls print()

l2 = Leveltwo(4.0) # calls initialise()
a.dtype = l2 # calls set()
a.dtype.print_() # calls print()
