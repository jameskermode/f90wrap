from __future__ import print_function, absolute_import, division
import _test_collision
import f90wrap.runtime
import logging
import numpy
import warnings
import weakref

class Test_Module(f90wrap.runtime.FortranModule):
    """
    Module test_module
    Defined at test_module.f90 lines 3-23
    """
    @staticmethod
    def enable_timing(interface_call=False):
        """
        enable_timing()
        Defined at test_module.f90 lines 7-8
        
        """
        _test_collision.f90wrap_test_module__enable_timing()
    
    @staticmethod
    def system_init(enable_timing, verbosity=None, interface_call=False):
        """
        system_init(enable_timing[, verbosity])
        Defined at test_module.f90 lines 13-23
        
        Parameters
        ----------
        enable_timing : bool
        verbosity : int32
        """
        _test_collision.f90wrap_test_module__system_init(enable_timing=enable_timing, \
            verbosity=verbosity)
    
    _dt_array_initialisers = []
    

test_module = Test_Module()

