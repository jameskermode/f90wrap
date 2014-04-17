import functools
import f90wrap.arraydata
import pysrc

class Leveltwo(object):
    @functools.wraps(pysrc.f90wrap_leveltwo_initialise, assigned=['__doc__'])
    def __init__(self, rl=None, handle=None):
        self._alloc = handle is None
        if self._alloc:
            handle = pysrc.f90wrap_leveltwo_initialise(rl)
        self._handle = handle

    def __del__(self):
        if self._alloc:
            pysrc.f90wrap_leveltwo_finalise(self._handle)
    
    @property
    def rl(self):
        return pysrc.f90wrap_leveltwo__get__rl(self._handle)

    @rl.setter
    def rl(self, rl):
        pysrc.f90wrap_leveltwo__set__rl(self._handle, rl)

    @functools.wraps(pysrc.f90wrap_leveltwo_print, assigned=['__doc__'])
    def print_(self):
        pysrc.f90wrap_leveltwo_print(self._handle)


class Atype(object):
    @functools.wraps(pysrc.f90wrap_atype_initialise, assigned=['__doc__'])    
    def __init__(self, handle=None):
        self._alloc = handle is None
        if self._alloc:
            handle = pysrc.f90wrap_atype_initialise()
        self._handle = handle
        self._objs = {}
        self._arrays = {}

    def __del__(self):
        if self._alloc:
            pysrc.f90wrap_atype_finalise(self._handle)

    @property
    def bool(self):
        return pysrc.f90wrap_atype__get__bool(self._handle)

    @bool.setter
    def bool(self, bool):
        pysrc.f90wrap_atype__set__bool(self._handle, bool)

    @property
    def dtype(self):
        dtype_handle = pysrc.f90wrap_atype__get__dtype(self._handle)
        if tuple(dtype_handle) in self._objs:
            dtype = self._objs[tuple(dtype_handle)]
        else:
            dtype = Leveltwo(handle=dtype_handle)
            self._objs[tuple(dtype_handle)] = dtype
        return dtype

    @dtype.setter
    def dtype(self, dtype):
        dtype = dtype._handle
        pysrc.f90wrap_atype__set__dtype(self._handle, dtype)

    @property
    def integ(self):
        return pysrc.f90wrap_atype__get__integ(self._handle)

    @integ.setter
    def integ(self, integ):
        pysrc.f90wrap_atype__set__integ(self._handle, integ)

    @property
    def rl(self):
        return pysrc.f90wrap_atype__get__rl(self._handle)

    @rl.setter
    def rl(self, rl):
        pysrc.f90wrap_atype__set__rl(self._handle, rl)

    @property
    def vec(self):
        if 'vec' in self._arrays:
            vec = self._arrays['vec']
        else:
            vec = f90wrap.arraydata.get_array(len(self._handle),
                                              self._handle,
                                              pysrc.f90wrap_atype__array__vec)
            self._arrays['vec'] = vec
        return vec

    @vec.setter
    def vec(self, vec):
        self.vec[:] = vec

@functools.wraps(pysrc.f90wrap_do_stuff, assigned=['__doc__'])
def do_stuff(factor):
    pysrc.f90wrap_do_stuff(factor)
