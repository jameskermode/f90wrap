"""
Module myclass_factory


Defined at myclass_factory.f90 lines 1-15

"""
from __future__ import print_function, absolute_import, division
import _itest
import f90wrap.runtime
import logging
import numpy

_arrays = {}
_objs = {}

def create_myclass(impl_type):
    """
    myobject = create_myclass(impl_type)
    
    
    Defined at myclass_factory.f90 lines 6-15
    
    Parameters
    ----------
    impl_type : str
    
    Returns
    -------
    myobject : Myclass_Impl_T
    
    """
    myobject = _itest.f90wrap_myclass_factory__create_myclass(impl_type=impl_type)
    myobject = \
        f90wrap.runtime.lookup_class("itest.myclass_impl_t").from_handle(myobject, \
        alloc=True)
    return myobject


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module \
        "myclass_factory".')

for func in _dt_array_initialisers:
    func()
