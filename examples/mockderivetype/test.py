import define_a_type as dt
import leveltwomod as l2
import use_a_type as ut

a = dt.Atype() # calls initialise()

a.rl = 3.0 # calls set()
print 'a.rl =', a.rl # calls get()

a.vec[:] = 0. # calls get() then sets array data in place
print 'a.vec =', a.vec # calls get()
a.vec = 1. # calls set()
print 'a.vec =', a.vec # calls get()

a.dtype.rl = 1.0 # calls set()

my_l2 = l2.Leveltwo(4.0) # calls initialise()
a.dtype = my_l2 # calls set()

# access the module-level variables in use_a_type
ut.fmod.p.rl = 1.0
ut.fmod.p.bool = True
ut.fmod.p.integ = 10

result = ut.do_stuff(8)
print 'result =', result
print 'ut.fmod.vector', ut.fmod.vector
