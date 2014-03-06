import pysrc

# import the wrapped module
print 'dir(pysrc)'
print dir(pysrc)

# allocate some instances of derived type variables
a_handle = pysrc.f90wrap_atype_initialise()
l2_handle = pysrc.f90wrap_leveltwo_initialise(3.0)

# call a function inside leveltwomod
pysrc.f90wrap_leveltwo_print(l2_handle)

# clear up and deallocate
pysrc.f90wrap_atype_finalise(a_handle)
pysrc.f90wrap_leveltwo_finalise(l2_handle)
