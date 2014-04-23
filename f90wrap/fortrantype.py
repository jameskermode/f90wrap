class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class FortranModule(object):
    __metaclass__ = Singleton
    def __init__(self):
        self._arrays = {}
        self._objs = {}        

class FortranDerivedType(object):
    def __init__(self):
        self._handle = None
        self._arrays = {}
        self._objs = {}
        self._alloc = True

    @classmethod
    def from_handle(cls, handle):
        self = cls.__new__(cls)
        FortranDerivedType.__init__(self) # always call the base constructor only
        self._handle = handle
        self._alloc = False
        return self
        
        

