# Changelog

## [Unreleased]

### Added
- Direct-C code generation support via `--direct-c` flag. Generates C extension modules that call f90wrap Fortran helpers, eliminating f2py dependency for supported code patterns.

### Implementation
- `f90wrap/directc.py`: Interop analysis and procedure classification
- `f90wrap/directc_cgen.py`: C code generator for Python C API wrappers
- `f90wrap/numpy_utils.py`: NumPy C API type mapping utilities
- CLI: `--direct-c` flag generates `_module.c` files alongside standard wrappers

### Notes
- Helpers-only path: all procedures call existing `f90wrap_<module>__<proc>` helpers
- Normal f2py workflow unchanged when `--direct-c` not specified
- Generated C files must be compiled separately with user's toolchain
