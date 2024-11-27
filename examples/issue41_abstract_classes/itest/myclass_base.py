"""
Module myclass_base


Defined at myclass_base.f90 lines 1-12

"""
from __future__ import print_function, absolute_import, division
import _itest
import f90wrap.runtime
import logging
import numpy

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("itest.myclass_t")
class myclass_t(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=myclass_t)


    Defined at myclass_base.f90 lines 3-5

    """
    def __init__(self, handle=None):
        """
        self = Myclass_T()


        Defined at myclass_base.f90 lines 3-5


        Returns
        -------
        this : Myclass_T
        	Object to be constructed


        Automatically generated constructor for myclass_t
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _itest.f90wrap_myclass_base__myclass_t_initialise()
        self._handle = result[0] if isinstance(result, tuple) else result

    def __del__(self):
        """
        Destructor for class Myclass_T


        Defined at myclass_base.f90 lines 3-5

        Parameters
        ----------
        this : Myclass_T
        	Object to be destructed


        Automatically generated destructor for myclass_t
        """
        if self._alloc:
            _itest.f90wrap_myclass_base__myclass_t_finalise(this=self._handle)

    def get_value(self):
        """
        value = get_value__binding__myclass_impl_t(self)


        Defined at myclass_base.f90 lines 10-13

        Parameters
        ----------
        self : Myclass_T

        Returns
        -------
        value : float

        """
        value = \
            _itest.f90wrap_myclass_base__get_value__binding__myclass_t(self=self._handle)
        return value

    _dt_array_initialisers = []



_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module \
        "myclass_base".')

for func in _dt_array_initialisers:
    func()
