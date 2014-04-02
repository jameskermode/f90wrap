import pysrc

# import the wrapped module
print 'dir(pysrc)'
print dir(pysrc)

# allocate some instances of derived type variables
a_handle = pysrc.f90wrap_atype_initialise()
l2_handle = pysrc.f90wrap_leveltwo_initialise(3.0)

# call a function inside leveltwomod
pysrc.f90wrap_leveltwo_print(l2_handle)

print pysrc.f90wrap_leveltwo__get__rl(l2_handle)
pysrc.f90wrap_leveltwo__set__rl(l2_handle, 2.)
pysrc.f90wrap_leveltwo__get__rl(l2_handle)

# get an array pointer
import f90wrap.arraydata
from f90wrap.sizeof_fortran_t import sizeof_fortran_t
vec = f90wrap.arraydata.get_array(sizeof_fortran_t(),
                                  a_handle,
                                  pysrc.f90wrap_atype__array__vec)
vec[:] = 0.
print vec

# clear up and deallocate
pysrc.f90wrap_atype_finalise(a_handle)
pysrc.f90wrap_leveltwo_finalise(l2_handle)
