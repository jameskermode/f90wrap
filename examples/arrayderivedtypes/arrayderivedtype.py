from __future__ import print_function, absolute_import, division
import _arrayderivedtype
import f90wrap.runtime
import logging
import numpy
import warnings

class Module_Calcul(f90wrap.runtime.FortranModule):
    """
    Module module_calcul
    Defined at test.fpp lines 5-16
    """
    @f90wrap.runtime.register_class("arrayderivedtype.type_ptmes")
    class type_ptmes(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=type_ptmes)
        Defined at test.fpp lines 6-7
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for type_ptmes
            
            self = Type_Ptmes()
            Defined at test.fpp lines 6-7
            
            Returns
            -------
            this : Type_Ptmes
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            result = _arrayderivedtype.f90wrap_module_calcul__type_ptmes_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
        
        def __del__(self):
            """
            Automatically generated destructor for type_ptmes
            
            Destructor for class Type_Ptmes
            Defined at test.fpp lines 6-7
            
            Parameters
            ----------
            this : Type_Ptmes
                Object to be destructed
            
            """
            if self._alloc:
                _arrayderivedtype.f90wrap_module_calcul__type_ptmes_finalise(this=self._handle)
        
        @property
        def y(self):
            """
            Element y ftype=integer  pytype=int
            Defined at test.fpp line 7
            """
            return _arrayderivedtype.f90wrap_type_ptmes__get__y(self._handle)
        
        @y.setter
        def y(self, y):
            _arrayderivedtype.f90wrap_type_ptmes__set__y(self._handle, y)
        
        def __str__(self):
            ret = ['<type_ptmes>{\n']
            ret.append('    y : ')
            ret.append(repr(self.y))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("arrayderivedtype.array_type")
    class array_type(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=array_type)
        Defined at test.fpp lines 9-10
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for array_type
            
            self = Array_Type()
            Defined at test.fpp lines 9-10
            
            Returns
            -------
            this : Array_Type
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            result = _arrayderivedtype.f90wrap_module_calcul__array_type_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
        
        def __del__(self):
            """
            Automatically generated destructor for array_type
            
            Destructor for class Array_Type
            Defined at test.fpp lines 9-10
            
            Parameters
            ----------
            this : Array_Type
                Object to be destructed
            
            """
            if self._alloc:
                _arrayderivedtype.f90wrap_module_calcul__array_type_finalise(this=self._handle)
        
        def init_array_x(self):
            self.x = f90wrap.runtime.FortranDerivedTypeArray(self,
                                            _arrayderivedtype.f90wrap_array_type__array_getitem__x,
                                            _arrayderivedtype.f90wrap_array_type__array_setitem__x,
                                            _arrayderivedtype.f90wrap_array_type__array_len__x,
                                            """
            Element x ftype=type(type_ptmes) pytype=Type_Ptmes
            Defined at test.fpp line 10
            """, Module_Calcul.type_ptmes)
            return self.x
        
        _dt_array_initialisers = [init_array_x]
        
    
    @staticmethod
    def recup_point(self, interface_call=False):
        """
        recup_point(self)
        Defined at test.fpp lines 14-16
        
        Parameters
        ----------
        x : Array_Type
        """
        _arrayderivedtype.f90wrap_module_calcul__recup_point(x=self._handle)
    
    def init_array_xarr(self):
        self.xarr = f90wrap.runtime.FortranDerivedTypeArray(f90wrap.runtime.empty_type,
                                        _arrayderivedtype.f90wrap_module_calcul__array_getitem__xarr,
                                        _arrayderivedtype.f90wrap_module_calcul__array_setitem__xarr,
                                        _arrayderivedtype.f90wrap_module_calcul__array_len__xarr,
                                        """
        Element xarr ftype=type(array_type) pytype=Array_Type
        Defined at test.fpp line 12
        """, Module_Calcul.array_type)
        return self.xarr
    
    _dt_array_initialisers = [init_array_xarr]
    

module_calcul = Module_Calcul()

