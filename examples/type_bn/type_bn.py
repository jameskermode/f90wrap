import _type_bn
import f90wrap.runtime
import logging

class Module_Structure(f90wrap.runtime.FortranModule):
    """
    Module module_structure
    
    
    Defined at type_bn.f90 lines 1-5
    
    """
    class Type_Face(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=type_face)
        
        
        Defined at type_bn.f90 lines 3-5
        
        """
        def __init__(self, handle=None):
            """
            self = Type_Face()
            
            
            Defined at type_bn.f90 lines 3-5
            
            
            Returns
            -------
            this : Type_Face
            	Object to be constructed
            
            
            Automatically generated constructor for type_face
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            self._handle = _type_bn.f90wrap_type_face_initialise()
        
        def __del__(self):
            """
            Destructor for class Type_Face
            
            
            Defined at type_bn.f90 lines 3-5
            
            Parameters
            ----------
            this : Type_Face
            	Object to be destructed
            
            
            Automatically generated destructor for type_face
            """
            if self._alloc:
                _type_bn.f90wrap_type_face_finalise(this=self._handle)
        
        @property
        def type(self):
            """
            Element type ftype=integer  pytype=int
            
            
            Defined at type_bn.f90 line 4
            
            """
            return _type_bn.f90wrap_type_face__get__type(self._handle)
        
        @type.setter
        def type(self, type):
            _type_bn.f90wrap_type_face__set__type(self._handle, type)
        
        def __str__(self):
            ret = ['<type_face>{\n']
            ret.append('    type : ')
            ret.append(repr(self.type))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    _dt_array_initialisers = []
    

module_structure = Module_Structure()

