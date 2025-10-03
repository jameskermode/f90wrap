from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings

class M_String_Test(f90wrap.runtime.FortranModule):
    """
    Module m_string_test
    Defined at main.f90 lines 1-60
    """
    @staticmethod
    def string_out_optional_array(output=None, interface_call=False):
        """
        string_out_optional_array([output])
        Defined at main.f90 lines 56-60
        
        Parameters
        ----------
        output : str array
        """
        if output is not None:
            if isinstance(output,(numpy.ndarray, numpy.generic)):
                if output.ndim not in {1,2} or output.dtype.num not in {18,19,2}:
                    raise TypeError("Expecting 'str' (code '2')"
                    " with dim '1' but got '%s' (code '%s') with dim '%s'"
                    %(output.dtype, output.dtype.num, output.ndim))
            else:
                raise TypeError("Expecting numpy array but got '%s'"%type(output))
        _pywrapper.f90wrap_m_string_test__string_out_optional_array(output=output)
    
    @staticmethod
    def string_in_array(input, interface_call=False):
        """
        string_in_array(input)
        Defined at main.f90 lines 16-24
        
        Parameters
        ----------
        input : str array
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if input.ndim not in {1,2} or input.dtype.num not in {18,19,2}:
                raise TypeError("Expecting 'str' (code '2')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(input))
        _pywrapper.f90wrap_m_string_test__string_in_array(input=input)
    
    @staticmethod
    def string_in(input, interface_call=False):
        """
        string_in(input)
        Defined at main.f90 lines 13-14
        
        Parameters
        ----------
        input : str
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if not interface_call and input.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                12, 13}:
                input = input.astype('uint8')
            if input.ndim not in {0} or input.dtype.num not in {18,19,2}:
                raise TypeError("Expecting 'str' (code '2')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        elif not isinstance(input,str):
            raise TypeError("Expecting 'str' but got '%s'"%type(input))
        _pywrapper.f90wrap_m_string_test__string_in(input=input)
    
    @staticmethod
    def string_in_array_hardcoded_size(input, interface_call=False):
        """
        string_in_array_hardcoded_size(input)
        Defined at main.f90 lines 26-34
        
        Parameters
        ----------
        input : str array
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if input.dtype.num not in {18,19,2}:
                raise TypeError("Expecting 'str' (code '2')"
                " with dim '-1' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(input))
        _pywrapper.f90wrap_m_string_test__string_in_array_hardcoded_size(input=input)
    
    @staticmethod
    def string_to_string(input, interface_call=False):
        """
        output = string_to_string(input)
        Defined at main.f90 lines 36-39
        
        Parameters
        ----------
        input : str
        
        Returns
        -------
        output : str
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if not interface_call and input.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                12, 13}:
                input = input.astype('uint8')
            if input.ndim not in {0} or input.dtype.num not in {18,19,2}:
                raise TypeError("Expecting 'str' (code '2')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        elif not isinstance(input,str):
            raise TypeError("Expecting 'str' but got '%s'"%type(input))
        output = _pywrapper.f90wrap_m_string_test__string_to_string(input=input)
        return output
    
    @staticmethod
    def string_out_optional(output=None, interface_call=False):
        """
        string_out_optional([output])
        Defined at main.f90 lines 50-54
        
        Parameters
        ----------
        output : str
        """
        if output is not None:
            if isinstance(output,(numpy.ndarray, numpy.generic)):
                if output.ndim not in {0} or output.dtype.num not in {18,19,2}:
                    raise TypeError("Expecting 'str' (code '2')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(output.dtype, output.dtype.num, output.ndim))
            elif not isinstance(output,str):
                raise TypeError("Expecting 'str' but got '%s'"%type(output))
        _pywrapper.f90wrap_m_string_test__string_out_optional(output=output)
    
    @staticmethod
    def string_out(interface_call=False):
        """
        output = string_out()
        Defined at main.f90 lines 46-48
        
        Returns
        -------
        output : str
        """
        output = _pywrapper.f90wrap_m_string_test__string_out()
        return output
    
    @staticmethod
    def string_to_string_array(input, output, interface_call=False):
        """
        string_to_string_array(input, output)
        Defined at main.f90 lines 41-44
        
        Parameters
        ----------
        input : str array
        output : str array
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if input.ndim not in {1,2} or input.dtype.num not in {18,19,2}:
                raise TypeError("Expecting 'str' (code '2')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(input))
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim not in {1,2} or output.dtype.num not in {18,19,2}:
                raise TypeError("Expecting 'str' (code '2')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(output))
        _pywrapper.f90wrap_m_string_test__string_to_string_array(input=input, \
            output=output)
    
    _dt_array_initialisers = []
    

m_string_test = M_String_Test()

