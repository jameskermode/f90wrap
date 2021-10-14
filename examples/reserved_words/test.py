from __future__ import print_function, absolute_import, division
import _test
import f90wrap.runtime
import logging

class Highest_Level(f90wrap.runtime.FortranModule):
    """
    Module highest_level
    
    
    Defined at \
        /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
        lines 5-23
    
    """
    @f90wrap.runtime.register_class("test.Size_bn")
    class Size_bn(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=size_bn)
        
        
        Defined at \
            /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
            lines 7-11
        
        """
        def __init__(self, handle=None):
            """
            self = Size()
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                lines 7-11
            
            
            Returns
            -------
            this : Size
            	Object to be constructed
            
            
            Automatically generated constructor for size
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            result = _test.f90wrap_size_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
        
        def __del__(self):
            """
            Destructor for class Size
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                lines 7-11
            
            Parameters
            ----------
            this : Size
            	Object to be destructed
            
            
            Automatically generated destructor for size
            """
            if self._alloc:
                _test.f90wrap_size_finalise(this=self._handle)
        
        @property
        def test_double(self):
            """
            Element test_double ftype=real(kind=8) pytype=float
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                line 9
            
            """
            return _test.f90wrap_size__get__test_double(self._handle)
        
        @test_double.setter
        def test_double(self, test_double):
            _test.f90wrap_size__set__test_double(self._handle, test_double)
        
        @property
        def test_single(self):
            """
            Element test_single ftype=real(kind=4) pytype=float
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                line 10
            
            """
            return _test.f90wrap_size__get__test_single(self._handle)
        
        @test_single.setter
        def test_single(self, test_single):
            _test.f90wrap_size__set__test_single(self._handle, test_single)
        
        @property
        def test_float(self):
            """
            Element test_float ftype=real          pytype=float
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                line 11
            
            """
            return _test.f90wrap_size__get__test_float(self._handle)
        
        @test_float.setter
        def test_float(self, test_float):
            _test.f90wrap_size__set__test_float(self._handle, test_float)
        
        def __str__(self):
            ret = ['<size_bn>{\n']
            ret.append('    test_double : ')
            ret.append(repr(self.test_double))
            ret.append(',\n    test_single : ')
            ret.append(repr(self.test_single))
            ret.append(',\n    test_float : ')
            ret.append(repr(self.test_float))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("test.oktype")
    class oktype(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=oktype)
        
        
        Defined at \
            /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
            lines 13-15
        
        """
        def __init__(self, handle=None):
            """
            self = Oktype()
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                lines 13-15
            
            
            Returns
            -------
            this : Oktype
            	Object to be constructed
            
            
            Automatically generated constructor for oktype
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            result = _test.f90wrap_oktype_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
        
        def __del__(self):
            """
            Destructor for class Oktype
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                lines 13-15
            
            Parameters
            ----------
            this : Oktype
            	Object to be destructed
            
            
            Automatically generated destructor for oktype
            """
            if self._alloc:
                _test.f90wrap_oktype_finalise(this=self._handle)
        
        @property
        def test_single(self):
            """
            Element test_single ftype=real(kind=4) pytype=float
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                line 15
            
            """
            return _test.f90wrap_oktype__get__test_single(self._handle)
        
        @test_single.setter
        def test_single(self, test_single):
            _test.f90wrap_oktype__set__test_single(self._handle, test_single)
        
        def __str__(self):
            ret = ['<oktype>{\n']
            ret.append('    test_single : ')
            ret.append(repr(self.test_single))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("test.outer")
    class outer(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=outer)
        
        
        Defined at \
            /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
            lines 17-20
        
        """
        def __init__(self, handle=None):
            """
            self = Outer()
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                lines 17-20
            
            
            Returns
            -------
            this : Outer
            	Object to be constructed
            
            
            Automatically generated constructor for outer
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            result = _test.f90wrap_outer_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
        
        def __del__(self):
            """
            Destructor for class Outer
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                lines 17-20
            
            Parameters
            ----------
            this : Outer
            	Object to be destructed
            
            
            Automatically generated destructor for outer
            """
            if self._alloc:
                _test.f90wrap_outer_finalise(this=self._handle)
        
        @property
        def size_bn(self):
            """
            Element size_bn ftype=type(size) pytype=Size
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                line 19
            
            """
            size_bn_handle = _test.f90wrap_outer__get__size(self._handle)
            if tuple(size_bn_handle) in self._objs:
                size_bn = self._objs[tuple(size_bn_handle)]
            else:
                size_bn = highest_level.Size_bn.from_handle(size_bn_handle)
                self._objs[tuple(size_bn_handle)] = size_bn
            return size_bn
        
        @size_bn.setter
        def size_bn(self, size_bn):
            size_bn = size_bn._handle
            _test.f90wrap_outer__set__size_bn(self._handle, size_bn)
        
        @property
        def oktype(self):
            """
            Element oktype ftype=type(oktype) pytype=Oktype
            
            
            Defined at \
                /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
                line 20
            
            """
            oktype_handle = _test.f90wrap_outer__get__oktype(self._handle)
            if tuple(oktype_handle) in self._objs:
                oktype = self._objs[tuple(oktype_handle)]
            else:
                oktype = highest_level.oktype.from_handle(oktype_handle)
                self._objs[tuple(oktype_handle)] = oktype
            return oktype
        
        @oktype.setter
        def oktype(self, oktype):
            oktype = oktype._handle
            _test.f90wrap_outer__set__oktype(self._handle, oktype)
        
        def __str__(self):
            ret = ['<outer>{\n']
            ret.append('    size_bn : ')
            ret.append(repr(self.size_bn))
            ret.append(',\n    oktype : ')
            ret.append(repr(self.oktype))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @property
    def size_tmp(self):
        """
        Element size_tmp ftype=type(size) pytype=Size
        
        
        Defined at \
            /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
            line 22
        
        """
        size_tmp_handle = _test.f90wrap_highest_level__get__size_tmp()
        if tuple(size_tmp_handle) in self._objs:
            size_tmp = self._objs[tuple(size_tmp_handle)]
        else:
            size_tmp = highest_level.Size_bn.from_handle(size_tmp_handle)
            self._objs[tuple(size_tmp_handle)] = size_tmp
        return size_tmp
    
    @size_tmp.setter
    def size_tmp(self, size_tmp):
        size_tmp = size_tmp._handle
        _test.f90wrap_highest_level__set__size_tmp(size_tmp)
    
    @property
    def oktype_tmp(self):
        """
        Element oktype_tmp ftype=type(oktype) pytype=Oktype
        
        
        Defined at \
            /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
            line 23
        
        """
        oktype_tmp_handle = _test.f90wrap_highest_level__get__oktype_tmp()
        if tuple(oktype_tmp_handle) in self._objs:
            oktype_tmp = self._objs[tuple(oktype_tmp_handle)]
        else:
            oktype_tmp = highest_level.oktype.from_handle(oktype_tmp_handle)
            self._objs[tuple(oktype_tmp_handle)] = oktype_tmp
        return oktype_tmp
    
    @oktype_tmp.setter
    def oktype_tmp(self, oktype_tmp):
        oktype_tmp = oktype_tmp._handle
        _test.f90wrap_highest_level__set__oktype_tmp(oktype_tmp)
    
    @property
    def outer_tmp(self):
        """
        Element outer_tmp ftype=type(outer) pytype=Outer
        
        
        Defined at \
            /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/highest.fpp \
            line 24
        
        """
        outer_tmp_handle = _test.f90wrap_highest_level__get__outer_tmp()
        if tuple(outer_tmp_handle) in self._objs:
            outer_tmp = self._objs[tuple(outer_tmp_handle)]
        else:
            outer_tmp = highest_level.outer.from_handle(outer_tmp_handle)
            self._objs[tuple(outer_tmp_handle)] = outer_tmp
        return outer_tmp
    
    @outer_tmp.setter
    def outer_tmp(self, outer_tmp):
        outer_tmp = outer_tmp._handle
        _test.f90wrap_highest_level__set__outer_tmp(outer_tmp)
    
    def __str__(self):
        ret = ['<highest_level>{\n']
        ret.append('    size_tmp : ')
        ret.append(repr(self.size_tmp))
        ret.append(',\n    oktype_tmp : ')
        ret.append(repr(self.oktype_tmp))
        ret.append(',\n    outer_tmp : ')
        ret.append(repr(self.outer_tmp))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

highest_level = Highest_Level()

def invert(a, lda, n, rcond, determ):
    """
    invert(a, lda, n, rcond, determ)
    
    
    Defined at \
        /Users/ananthsridharan/codes/f90wrap/examples/reserved_words/cmake/build/bin/test/f90wrap/lapack_wrappers.fpp \
        lines 11-42
    
    Parameters
    ----------
    a : float array
    lda : int
    n : int
    rcond : float
    determ : float
    
    ======================================================================
     Inputs
    ======================================================================
    """
    _test.f90wrap_invert(a=a, lda=lda, n=n, rcond=rcond, determ=determ)

