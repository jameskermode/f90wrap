from __future__ import print_function, absolute_import, division
import _KeywordRename
import f90wrap.runtime
import logging

class Parameters(f90wrap.runtime.FortranModule):
    """
    Module parameters
    
    
    Defined at rename.fpp lines 5-9
    
    """
    @property
    def abc(self):
        """
        Element abc ftype=integer pytype=int
        
        
        Defined at rename.fpp line 7
        
        """
        return _KeywordRename.f90wrap_parameters__get__abc()
    
    @property
    def lambda_(self):
        """
        Element lambda_ ftype=integer pytype=int
        
        
        Defined at rename.fpp line 8
        
        """
        return _KeywordRename.f90wrap_parameters__get__lambda_()
    
    def __str__(self):
        ret = ['<parameters>{\n']
        ret.append('    abc : ')
        ret.append(repr(self.abc))
        ret.append(',\n    lambda_ : ')
        ret.append(repr(self.lambda_))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

parameters = Parameters()

def in_(a):
    """
    in_(a)
    
    
    Defined at rename.fpp lines 11-14
    
    Parameters
    ----------
    a : int
    
    """
    _KeywordRename.f90wrap_in_(a=a)

