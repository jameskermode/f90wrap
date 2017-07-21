from __future__ import print_function, absolute_import, division
import _example
import f90wrap.runtime
import logging

class Class_Example(f90wrap.runtime.FortranModule):
    """
    Module class_example
    
    
    Defined at ./example.f90 lines 2-73
    
    """
    @f90wrap.runtime.register_class("Example")
    class Example(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=example)
        
        
        Defined at ./example.f90 lines 12-16
        
        """
        def __init__(self, handle=None):
            """
            self = Example()
            
            
            Defined at ./example.f90 lines 12-16
            
            
            Returns
            -------
            this : Example
            	Object to be constructed
            
            
            Automatically generated constructor for example
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            self._handle = _example.f90wrap_example_initialise()
        
        def __del__(self):
            """
            Destructor for class Example
            
            
            Defined at ./example.f90 lines 12-16
            
            Parameters
            ----------
            this : Example
            	Object to be destructed
            
            
            Automatically generated destructor for example
            """
            if self._alloc:
                _example.f90wrap_example_finalise(this=self._handle)
        
        @property
        def first(self):
            """
            Element first ftype=integer  pytype=int
            
            
            Defined at ./example.f90 line 13
            
            """
            return _example.f90wrap_example__get__first(self._handle)
        
        @first.setter
        def first(self, first):
            _example.f90wrap_example__set__first(self._handle, first)
        
        @property
        def second(self):
            """
            Element second ftype=integer  pytype=int
            
            
            Defined at ./example.f90 line 14
            
            """
            return _example.f90wrap_example__get__second(self._handle)
        
        @second.setter
        def second(self, second):
            _example.f90wrap_example__set__second(self._handle, second)
        
        @property
        def third(self):
            """
            Element third ftype=integer  pytype=int
            
            
            Defined at ./example.f90 line 15
            
            """
            return _example.f90wrap_example__get__third(self._handle)
        
        @third.setter
        def third(self, third):
            _example.f90wrap_example__set__third(self._handle, third)
        
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
        
    
    @staticmethod
    def _return_example_first(first):
        """
        instance = _return_example_first(first)
        
        
        Defined at ./example.f90 lines 25-36
        
        Parameters
        ----------
        first : int
        
        Returns
        -------
        instance : Example
        
        """
        instance = _example.f90wrap_return_example_first(first=first)
        instance = f90wrap.runtime.lookup_class("Example").from_handle(instance)
        return instance
    
    @staticmethod
    def _return_example_second(first, second):
        """
        instance = _return_example_second(first, second)
        
        
        Defined at ./example.f90 lines 40-53
        
        Parameters
        ----------
        first : int
        second : int
        
        Returns
        -------
        instance : Example
        
        """
        instance = _example.f90wrap_return_example_second(first=first, second=second)
        instance = f90wrap.runtime.lookup_class("Example").from_handle(instance)
        return instance
    
    @staticmethod
    def _return_example_third(first, second, third):
        """
        instance = _return_example_third(first, second, third)
        
        
        Defined at ./example.f90 lines 57-72
        
        Parameters
        ----------
        first : int
        second : int
        third : int
        
        Returns
        -------
        instance : Example
        
        """
        instance = _example.f90wrap_return_example_third(first=first, second=second, \
            third=third)
        instance = f90wrap.runtime.lookup_class("Example").from_handle(instance)
        return instance
    
    @staticmethod
    def return_example(*args, **kwargs):
        """
        return_example(*args, **kwargs)
        
        
        Defined at ./example.f90 lines 8-10
        
        Overloaded interface containing the following procedures:
          _return_example_first
          _return_example_second
          _return_example_third
        
        """
        for proc in [Class_Example._return_example_first, \
            Class_Example._return_example_second, Class_Example._return_example_third]:
            try:
                return proc(*args, **kwargs)
            except TypeError:
                continue
    
    _dt_array_initialisers = []
    

class_example = Class_Example()

