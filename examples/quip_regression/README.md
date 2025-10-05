# QUIP Regression Test

This directory contains a regression test that verifies f90wrap can successfully build QUIP/quippy, a real-world scientific computing project that uses f90wrap extensively.

## Purpose

QUIP (QUantum mechanics and Interatomic Potentials) is a large Fortran codebase with complex derived types, overloaded interfaces, and other advanced Fortran features. It serves as an excellent regression test for f90wrap because:

- It uses many f90wrap features in production
- It has complex Fortran code patterns
- Breaking changes in f90wrap will immediately show up as build failures
- It includes real-world use cases

## Requirements

- Python 3.9+
- OpenBLAS library (must be discoverable via pkg-config)
- meson and ninja build tools
- gfortran compiler
- git
- Required Python packages: numpy, ase, f90wrap

**Note**: The test will automatically skip if OpenBLAS is not available via pkg-config.

## Running the Test

### Standalone

```bash
python test_quip_build.py
```

This will:
1. Clone the QUIP repository (mesonify branch)
2. Build QUIP libraries with meson
3. Build the quippy Python package using f90wrap
4. Test basic quippy functionality

The test takes several minutes to run as it builds a large Fortran codebase.

### As part of f90wrap test suite

This test is intentionally **not** included in the regular f90wrap test suite run by `make test` because:
- It takes several minutes to complete (vs seconds for other tests)
- It requires a network connection to clone QUIP
- It requires significant disk space (>100 MB)

To include it in CI/CD, add it to a separate workflow or run it manually.

## Test Coverage

The test verifies:

1. **f90wrap wrapper generation**: QUIP contains many Fortran modules that f90wrap must process
2. **Complex derived types**: Tests f90wrap handling of nested types and type-bound procedures
3. **Overloaded interfaces**: Verifies the fix for overloaded interface shadowing bugs
4. **Dictionary types**: Tests integer/real/logical value storage and retrieval
5. **Build system integration**: Confirms meson-python integration works correctly

## Troubleshooting

### Test times out
Increase the timeout or run on a faster machine. The QUIP build can take 3-5 minutes.

### Build failures
Check that you have:
- Latest f90wrap installed: `pip install -e ../..` from the f90wrap root
- Required system packages: `gfortran`, `meson`, `ninja`
- Sufficient disk space (>500 MB for build artifacts)

### Import errors
The test installs quippy in editable mode. If imports fail, check that the installation completed successfully.

## Maintenance

This test targets the `mesonify` branch of QUIP which contains the meson-python build configuration. If this branch is merged or renamed, update `QUIP_BRANCH` in `test_quip_build.py`.

## Related

- QUIP repository: https://github.com/libAtoms/QUIP
- QUIP PR with f90wrap fixes: https://github.com/libAtoms/QUIP/pull/694
