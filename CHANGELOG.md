# Changelog

## [Unreleased]

### Added
- **--build flag**: One-step wrap and build for extension modules. Use `f90wrap --build -m module source.f90`. Automatically uses Direct-C mode. For f2py or complex builds, continue using Makefiles.
- **Direct-C mode**: Alternative to f2py for generating Python extension modules. Use `--direct-c` flag to generate C code that directly calls f90wrap Fortran helpers via the Python C API, eliminating the f2py dependency.

### Improved
- **More reliable cleanup**: Switched from `__del__` to `weakref.finalize` for derived type destructors. This fixes cleanup in reference cycles, improves thread-safety, and provides deterministic finalization order. Recommended Python best practice since 3.4.

### Implementation
- `f90wrap/build.py`: Build orchestration module with clean API for both CLI and programmatic use
- `f90wrap/setuptools_ext.py`: Setuptools integration with F90WrapExtension and build_ext_cmdclass
- `f90wrap/directc.py`: ISO C interoperability analysis and procedure classification
- `f90wrap/directc_cgen/`: C code generator package for Python C API wrappers
- `f90wrap/numpy_utils.py`: NumPy C API type mapping utilities
- `f90wrap/runtime.py`: Runtime support for Direct-C array handling
- CLI: `--build` and `--clean-build` flags for automated compilation
- CLI: `--direct-c` flag generates `_module.c` files alongside standard Fortran wrappers

### Build System Details
- Respects standard environment variables: `FC`, `F90`, `CC`, `FFLAGS`, `CFLAGS`, `LDFLAGS`, `F2PY`, `F2PY_F90WRAP`
- Auto-detects Python and NumPy include paths (overridable via `PYTHON_INCLUDES`, `NUMPY_INCLUDES`)
- Platform-specific defaults (Darwin bundle vs Linux shared library)
- Compiles f90wrap-generated wrappers and links with specified source files
- Programmatic API for Python package builds (pyproject.toml/setup.py integration)
- For complex builds with external libraries, continue using Makefiles or build systems

### Direct-C Mode Details
- Generates standalone C extension modules using Python C API
- All procedures call existing `f90wrap_<module>__<proc>` Fortran helpers
- Works with `--build` flag for one-step generation and compilation
- Normal f2py workflow unchanged when `--direct-c` not specified
- See README.md for complete usage instructions
