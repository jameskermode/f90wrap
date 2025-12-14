# Issue #306 Analysis: Module-level allocatable arrays fail after reallocation

## Root Cause

The issue is a **mismatch between `sizeof_fortran_t` at wrapper generation time vs runtime**.

### How it happens

1. When f90wrap generates Fortran wrappers, it hardcodes `dummy_this(N)` where N is `sizeof_fortran_t` computed on the **generation** system
2. At runtime, f90wrap computes `sizeof_fortran_t` again and creates `empty_handle = [0]*sizeof_fortran_t`
3. If these values differ, f2py throws: "0-th dimension must be fixed to X but got Y"

### Reporter's Environment

Error message: "0-th dimension must be fixed to 2 but got 4"

This indicates:
- GCC 15.2.1 produces `sizeof_fortran_t = 2`
- The wrappers were generated on a system with `sizeof_fortran_t = 4`

### Reproduction

On a system with `sizeof_fortran_t = 4`, passing a size-2 handle reproduces the error:

```python
>>> import _alloc_mod
>>> _alloc_mod.f90wrap_alloc_mod__array__data_array([0, 0])  # size 2 handle
ValueError: 0-th dimension must be fixed to 4 but got 2
```

### Why GCC 15?

`sizeof_fortran_t` is computed in `f90wrap/sizeoffortran.f90` by measuring the size of a Fortran derived type pointer using `transfer()`. GCC 15 may have changed the internal representation of type/class pointers, reducing their size from 4 integers to 2.

### Key Observation

For **module-level** arrays, `dummy_this` is actually **unused**! The compiler warns:

```
Warning: Unused dummy argument 'dummy_this' at (1) [-Wunused-dummy-argument]
```

Looking at the generated Fortran wrapper:
```fortran
subroutine f90wrap_alloc_mod__array__data_array(dummy_this, nd, dtype, dshape, dloc)
    integer, intent(in) :: dummy_this(4)  ! UNUSED - never referenced
    ! ... dummy_this is never used in the body
end subroutine
```

## Proposed Fix

For module-level array accessors, we should either:

1. **Remove `dummy_this` entirely** - since it's unused for modules
2. **Make the size dynamic** - use assumed-size array `dummy_this(*)`

Option 1 is cleaner but requires changes to both:
- `f90wrap/f90wrapgen.py` - Fortran wrapper generation
- `f90wrap/pywrapgen.py` - Python wrapper generation

Option 2 is simpler but may have f2py compatibility issues.

## Files to Modify

- `f90wrap/f90wrapgen.py` - generates the Fortran `f90wrap_*__array__*` subroutines
- `f90wrap/pywrapgen.py` - generates the Python property that calls these subroutines

## Test Case

The test case in this directory (`alloc_mod.f90`, `tests.py`) passes on systems where `sizeof_fortran_t` matches between generation and runtime (GCC 13/14). It would fail on GCC 15 due to the size mismatch.

## Questions for Reporter

1. Confirm `sizeof_fortran_t` value:
   ```python
   from f90wrap.sizeof_fortran_t import sizeof_fortran_t
   print(sizeof_fortran_t())
   ```

2. How was f90wrap installed? (PyPI wheels vs source build)
