from mockdt import (define_a_type,
                    leveltwomod,
                    use_a_type,
                    top_level)

a = define_a_type.Atype() # calls initialise()

a.rl = 3.0 # calls set()
print 'a.rl =', a.rl # calls get()

a.vec[:] = 0. # calls get() then sets array data in place
print 'a.vec =', a.vec # calls get()
a.vec = 1. # calls set()
print 'a.vec =', a.vec # calls get()

a.dtype.rl = 1.0 # calls set()

my_l2 = leveltwomod.Leveltwo(4.0) # calls initialise()
a.dtype = my_l2 # calls set()

# access the module-level variables in use_a_type
use_a_type.p.rl = 1.0
use_a_type.p.bool = True
use_a_type.p.integ = 10

result = use_a_type.do_stuff(8)
print 'result =', result
print 'use_a_type.vector', use_a_type.vector

input = 3.0
output = top_level(input)
print 'top_level(%.1f) = %.1f' % (input, output)
