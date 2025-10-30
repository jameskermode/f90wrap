"""
Module class_example
Defined at ./example.f90 lines 2-49
"""
from __future__ import print_function, absolute_import, division
import _examplepkg
import f90wrap.runtime
import logging
import numpy
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("examplepkg.Example")
class Example(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=example)
    Defined at ./example.f90 lines 9-12
    """
    def __init__(self, handle=None):
        """
        Automatically generated constructor for example
        
        self = Example()
        Defined at ./example.f90 lines 9-12
        
        Returns
        -------
        this : Example
            Object to be constructed
        
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        if handle is not None:
            self._handle = handle
            self._alloc = True
        else:
            result = _examplepkg.f90wrap_class_example__example_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
            self._alloc = True
    
    def __del__(self):
        """
        Automatically generated destructor for example
        
        Destructor for class Example
        Defined at ./example.f90 lines 9-12
        
        Parameters
        ----------
        this : Example
            Object to be destructed
        
        """
        if getattr(self, '_alloc', False):
            _examplepkg.f90wrap_class_example__example_finalise(this=self._handle)
    
    @property
    def first(self):
        """
        Element first ftype=integer  pytype=int
        Defined at ./example.f90 line 10
        """
        return _examplepkg.f90wrap_example__get__first(self._handle)
    
    @first.setter
    def first(self, first):
        _examplepkg.f90wrap_example__set__first(self._handle, first)
    
    @property
    def second(self):
        """
        Element second ftype=integer  pytype=int
        Defined at ./example.f90 line 11
        """
        return _examplepkg.f90wrap_example__get__second(self._handle)
    
    @second.setter
    def second(self, second):
        _examplepkg.f90wrap_example__set__second(self._handle, second)
    
    @property
    def third(self):
        """
        Element third ftype=integer  pytype=int
        Defined at ./example.f90 line 12
        """
        return _examplepkg.f90wrap_example__get__third(self._handle)
    
    @third.setter
    def third(self, third):
        _examplepkg.f90wrap_example__set__third(self._handle, third)
    
    def __str__(self):
        ret = ['<example>{\n']
        ret.append('    first : ')
        ret.append(repr(self.first))
        ret.append(',\n    second : ')
        ret.append(repr(self.second))
        ret.append(',\n    third : ')
        ret.append(repr(self.third))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

def return_example_third(first, second, third, interface_call=False):
    """
    instance = return_example_third(first, second, third)
    Defined at ./example.f90 lines 38-48
    
    Parameters
    ----------
    first : int32
    second : int32
    third : int32
    
    Returns
    -------
    instance : Example
    """
    instance = _examplepkg.f90wrap_class_example__return_example_third(first=first, \
        second=second, third=third)
    instance = \
        f90wrap.runtime.lookup_class("examplepkg.Example").from_handle(instance, \
        alloc=True)
    return instance

def return_example_second(first, second, interface_call=False):
    """
    instance = return_example_second(first, second)
    Defined at ./example.f90 lines 27-35
    
    Parameters
    ----------
    first : int32
    second : int32
    
    Returns
    -------
    instance : Example
    """
    instance = _examplepkg.f90wrap_class_example__return_example_second(first=first, \
        second=second)
    instance = \
        f90wrap.runtime.lookup_class("examplepkg.Example").from_handle(instance, \
        alloc=True)
    return instance

def return_example_first(first, interface_call=False):
    """
    instance = return_example_first(first)
    Defined at ./example.f90 lines 18-24
    
    Parameters
    ----------
    first : int32
    
    Returns
    -------
    instance : Example
    """
    instance = _examplepkg.f90wrap_class_example__return_example_first(first=first)
    instance = \
        f90wrap.runtime.lookup_class("examplepkg.Example").from_handle(instance, \
        alloc=True)
    return instance

def return_example(*args, **kwargs):
    """
    return_example(*args, **kwargs)
    Defined at ./example.f90 lines 6-7
    
    Overloaded interface containing the following procedures:
      return_example_third
      return_example_second
      return_example_first
    """
    for proc in [return_example_third, return_example_second, return_example_first]:
        exception=None
        try:
            return proc(*args, **kwargs, interface_call=True)
        except (TypeError, ValueError, AttributeError, IndexError, \
            numpy.exceptions.ComplexWarning) as err:
            exception = "'%s: %s'" % (type(err).__name__, str(err))
            continue
    
    argTypes=[]
    for arg in args:
        try:
            argTypes.append("%s: dims '%s', type '%s',"
            " type code '%s'"
            %(str(type(arg)),arg.ndim, arg.dtype, arg.dtype.num))
        except AttributeError:
            argTypes.append(str(type(arg)))
    raise TypeError("Not able to call a version of "
        "return_example compatible with the provided args:"
        "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module \
        "class_example".')

for func in _dt_array_initialisers:
    func()
