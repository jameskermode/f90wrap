"""Build orchestration for f90wrap-generated extension modules.

This module provides both CLI and programmatic interfaces for compiling
f90wrap-generated wrappers into Python extension modules.

Environment Variables (with sane defaults):
    FC: Fortran compiler (default: gfortran)
    F90: Fortran 90 compiler (default: $FC)
    CC: C compiler (default: gcc)
    FFLAGS: Fortran compiler flags (default: -fPIC)
    CFLAGS: C compiler flags (default: -fPIC)
    LDFLAGS: Linker flags (default: platform-specific)
    F2PY: f2py command (default: f2py)
    F2PY_F90WRAP: f2py-f90wrap command (default: f2py-f90wrap)
    PYTHON_INCLUDES: Python include path (default: auto-detected)
    NUMPY_INCLUDES: NumPy include path (default: auto-detected)

Example programmatic usage:
    from f90wrap import build

    # Simple build
    build.build_extension('mymodule', ['source.f90'])

    # Direct-C build with custom compiler
    build.build_extension(
        'mymodule',
        ['source.f90'],
        direct_c=True,
        env={'FC': 'ifort', 'CC': 'icc'}
    )
"""

import os
import sys
import subprocess
import sysconfig
import platform
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def get_default_env() -> Dict[str, str]:
    """Return default environment variables for compilation.

    Returns:
        Dictionary of environment variables with sane defaults.
    """
    fc = os.environ.get('FC', 'gfortran')
    return {
        'FC': fc,
        'F90': os.environ.get('F90', fc),
        'CC': os.environ.get('CC', 'gcc'),
        'FFLAGS': os.environ.get('FFLAGS', '-fPIC'),
        'CFLAGS': os.environ.get('CFLAGS', '-fPIC'),
        'LDFLAGS': os.environ.get('LDFLAGS', _get_platform_ldflags()),
        'F2PY': os.environ.get('F2PY', 'f2py'),
        'F2PY_F90WRAP': os.environ.get('F2PY_F90WRAP', 'f2py-f90wrap'),
    }


def _get_platform_ldflags() -> str:
    """Get platform-specific default linker flags.

    Returns:
        Linker flags appropriate for the platform.
    """
    if platform.system() == 'Darwin':
        return '-bundle -undefined dynamic_lookup -lgfortran'
    return '-shared -lgfortran'


def get_python_includes(env: Optional[Dict[str, str]] = None) -> str:
    """Get Python include directory path.

    Args:
        env: Optional environment dict to check for PYTHON_INCLUDES override.

    Returns:
        Path to Python include directory.
    """
    if env and 'PYTHON_INCLUDES' in env:
        return env['PYTHON_INCLUDES']
    if 'PYTHON_INCLUDES' in os.environ:
        return os.environ['PYTHON_INCLUDES']
    return sysconfig.get_path('include')


def get_numpy_includes(env: Optional[Dict[str, str]] = None) -> str:
    """Get NumPy include directory path.

    Args:
        env: Optional environment dict to check for NUMPY_INCLUDES override.

    Returns:
        Path to NumPy include directory.
    """
    if env and 'NUMPY_INCLUDES' in env:
        return env['NUMPY_INCLUDES']
    if 'NUMPY_INCLUDES' in os.environ:
        return os.environ['NUMPY_INCLUDES']

    try:
        import numpy
        return numpy.get_include()
    except ImportError:
        logger.error("NumPy not found but required for Direct-C mode")
        raise


def find_generated_wrappers() -> List[Path]:
    """Find f90wrap-generated Fortran wrapper files.

    Returns:
        List of f90wrap_*.f90 file paths.
    """
    return [Path(f) for f in glob.glob('f90wrap_*.f90')]


def find_generated_c_files(module_name: str) -> List[Path]:
    """Find f90wrap-generated C extension files.

    Args:
        module_name: Name of the module.

    Returns:
        List of _*.c file paths for Direct-C mode.
    """
    pattern = f'_{module_name}*.c'
    return [Path(f) for f in glob.glob(pattern)]


def run_command(
    cmd: List[str],
    verbose: bool = False,
    env: Optional[Dict[str, str]] = None
) -> int:
    """Run a shell command with optional verbosity.

    Args:
        cmd: Command and arguments as list.
        verbose: If True, print command and output.
        env: Optional environment variables to merge with os.environ.

    Returns:
        Return code from command.
    """
    if verbose:
        logger.info(' '.join(cmd))

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    try:
        result = subprocess.run(
            cmd,
            env=full_env,
            capture_output=not verbose,
            text=True,
            check=False
        )
        if result.returncode != 0 and not verbose:
            logger.error(f"Command failed: {' '.join(cmd)}")
            if result.stderr:
                logger.error(result.stderr)
        return result.returncode
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        return 1


def compile_fortran_sources(
    source_files: List[str],
    compiler: str = 'gfortran',
    flags: str = '-fPIC',
    verbose: bool = False
) -> List[Path]:
    """Compile Fortran source files to object files.

    Args:
        source_files: List of Fortran source file paths.
        compiler: Fortran compiler command.
        flags: Compiler flags.
        verbose: Enable verbose output.

    Returns:
        List of generated object file paths.

    Raises:
        RuntimeError: If compilation fails.
    """
    object_files: List[Path] = []

    for src in source_files:
        src_path = Path(src)
        obj_path = src_path.with_suffix('.o')

        cmd = [compiler] + flags.split() + ['-c', str(src_path), '-o', str(obj_path)]
        ret = run_command(cmd, verbose)

        if ret != 0:
            raise RuntimeError(f"Failed to compile {src_path}")

        object_files.append(obj_path)

    return object_files


def compile_c_sources(
    c_files: List[Path],
    compiler: str = 'gcc',
    flags: str = '-fPIC',
    includes: Optional[List[str]] = None,
    verbose: bool = False
) -> List[Path]:
    """Compile C source files to object files.

    Args:
        c_files: List of C source file paths.
        compiler: C compiler command.
        flags: Compiler flags.
        includes: List of include directory paths.
        verbose: Enable verbose output.

    Returns:
        List of generated object file paths.

    Raises:
        RuntimeError: If compilation fails.
    """
    object_files: List[Path] = []
    includes = includes or []

    for src in c_files:
        obj_path = src.with_suffix('.o')

        cmd = [compiler] + flags.split()
        for inc in includes:
            cmd.extend(['-I', inc])
        cmd.extend(['-c', str(src), '-o', str(obj_path)])

        ret = run_command(cmd, verbose)

        if ret != 0:
            raise RuntimeError(f"Failed to compile {src}")

        object_files.append(obj_path)

    return object_files


def link_shared_library(
    object_files: List[Path],
    output_name: str,
    linker: str = 'gcc',
    flags: str = '',
    verbose: bool = False
) -> Path:
    """Link object files into a shared library.

    Args:
        object_files: List of object file paths to link.
        output_name: Output shared library name (e.g., '_mymodule.so').
        linker: Linker command (usually gcc or compiler).
        flags: Linker flags.
        verbose: Enable verbose output.

    Returns:
        Path to generated shared library.

    Raises:
        RuntimeError: If linking fails.
    """
    output = Path(output_name)

    cmd = [linker]
    if flags:
        cmd.extend(flags.split())
    cmd.extend(['-o', str(output)] + [str(obj) for obj in object_files])

    ret = run_command(cmd, verbose)

    if ret != 0:
        raise RuntimeError(f"Failed to link {output}")

    return output


def build_with_f2py(
    module_name: str,
    source_files: List[str],
    env: Optional[Dict[str, str]] = None,
    verbose: bool = False
) -> int:
    """Build extension using f2py or f2py-f90wrap.

    Args:
        module_name: Name of the Python module.
        source_files: List of original Fortran source files (informational).
        env: Optional environment variables.
        verbose: Enable verbose output.

    Returns:
        0 on success, non-zero on failure.
    """
    env = env or {}
    defaults = get_default_env()
    full_env = {**defaults, **env}

    wrapper_files = find_generated_wrappers()
    if not wrapper_files:
        logger.error("No f90wrap_*.f90 files found. Run f90wrap first.")
        return 1

    logger.info("Compiling with f2py...")

    f2py_cmd = full_env['F2PY_F90WRAP']
    fc = full_env['F90']

    fortran_sources: List[str] = []
    for pattern in ['*.f90', '*.F90', '*.f', '*.F']:
        for f in glob.glob(pattern):
            if not f.startswith('f90wrap_') and f not in fortran_sources:
                fortran_sources.append(f)

    cmd = [
        f2py_cmd,
        '-c',
        '-m', f'_{module_name}',
        '--build-dir', 'build'
    ]

    if fc != 'gfortran':
        fcompiler_map = {
            'ifort': 'intelem',
            'ifx': 'intelem',
        }
        fcompiler = fcompiler_map.get(fc, fc)
        cmd.extend(['--fcompiler', fcompiler])

    cmd.extend([str(f) for f in wrapper_files])
    cmd.extend(fortran_sources)

    ret = run_command(cmd, verbose, full_env)

    if ret != 0:
        logger.error("f2py compilation failed")
        return ret

    logger.info(f"Successfully built _{module_name}.so")
    return 0


def build_direct_c(
    module_name: str,
    source_files: List[str],
    env: Optional[Dict[str, str]] = None,
    verbose: bool = False
) -> int:
    """Build extension using Direct-C compilation.

    Args:
        module_name: Name of the Python module.
        source_files: List of Fortran source files (may be .fpp preprocessed).
        env: Optional environment variables.
        verbose: Enable verbose output.

    Returns:
        0 on success, non-zero on failure.
    """
    env = env or {}
    defaults = get_default_env()
    full_env = {**defaults, **env}

    wrapper_files = find_generated_wrappers()
    if not wrapper_files:
        logger.error("No f90wrap_*.f90 files found. Run f90wrap first.")
        return 1

    c_files = find_generated_c_files(module_name)
    if not c_files:
        logger.error(f"No _{module_name}*.c files found. Use --direct-c flag.")
        return 1

    logger.info("Building with Direct-C mode...")

    try:
        real_sources = []
        for src in source_files:
            if src.endswith('.fpp'):
                base = src[:-4]
                for ext in ['.f90', '.F90', '.f', '.F']:
                    candidate = base + ext
                    if Path(candidate).exists():
                        real_sources.append(candidate)
                        break
                else:
                    real_sources.append(src)
            else:
                real_sources.append(src)

        all_fortran = real_sources + [str(f) for f in wrapper_files]

        logger.info(f"Compiling Fortran sources ({len(all_fortran)} files)...")
        f_objects = compile_fortran_sources(
            all_fortran,
            compiler=full_env['F90'],
            flags=full_env['FFLAGS'],
            verbose=verbose
        )

        logger.info("Compiling C extension...")
        py_inc = get_python_includes(env)
        np_inc = get_numpy_includes(env)

        c_objects = compile_c_sources(
            c_files,
            compiler=full_env['CC'],
            flags=full_env['CFLAGS'],
            includes=[py_inc, np_inc],
            verbose=verbose
        )

        logger.info("Linking shared library...")
        for c_file in c_files:
            module_base = c_file.stem
            output_name = f"{module_base}.so"

            needed_objects = []
            for obj in f_objects:
                base = obj.stem
                if base.startswith('f90wrap_'):
                    needed_objects.append(obj)
                elif any(Path(src).stem == base for src in real_sources):
                    needed_objects.append(obj)

            c_object = next((obj for obj in c_objects if obj.stem == module_base), None)
            if c_object:
                needed_objects.append(c_object)

            link_shared_library(
                needed_objects,
                output_name,
                linker=full_env['CC'],
                flags=full_env['LDFLAGS'],
                verbose=verbose
            )

            logger.info(f"Successfully built {output_name}")

        return 0

    except RuntimeError as exc:
        logger.error(str(exc))
        return 1


def build_extension(
    module_name: str,
    source_files: List[str],
    package_mode: bool = False,
    clean_first: bool = False,
    env: Optional[Dict[str, str]] = None,
    verbose: bool = False
) -> int:
    """Build f90wrap extension module after wrapper generation.

    This is the main entry point for both CLI and programmatic use.
    Uses Direct-C mode to compile wrappers and link extension modules.

    Args:
        module_name: Name of the Python module.
        source_files: List of Fortran source files to compile.
        package_mode: Package mode was used (-P flag).
        clean_first: Clean build artifacts before building.
        env: Optional environment variables to override defaults.
        verbose: Enable verbose output.

    Returns:
        0 on success, non-zero on failure.

    Example:
        >>> from f90wrap import build
        >>> build.build_extension('mymodule', ['src.f90', 'utils.f90'])
        0
    """
    if clean_first:
        clean_build_artifacts(module_name, package_mode)

    return build_direct_c(module_name, source_files, env, verbose)


def clean_build_artifacts(
    module_name: str,
    package_mode: bool = False,
    verbose: bool = False
):
    """Remove f90wrap-generated files and build artifacts.

    Args:
        module_name: Name of the module.
        package_mode: Whether package mode was used.
        verbose: Enable verbose output.
    """
    patterns = [
        'f90wrap_*.f90',
        'f90wrap_*.o',
        '_*.c',
        '_*.o',
        '_*.so',
        '*.mod',
        f'{module_name}.py',
        'build/',
        '__pycache__/',
        'src.*/',
        '.libs/',
        '.f2py_f2cmap',
    ]

    if package_mode:
        patterns.append(f'{module_name}/')

    for pattern in patterns:
        for path in glob.glob(pattern):
            path_obj = Path(path)
            try:
                if path_obj.is_dir():
                    import shutil
                    shutil.rmtree(path_obj)
                    if verbose:
                        logger.info(f"Removed directory: {path_obj}")
                else:
                    path_obj.unlink()
                    if verbose:
                        logger.info(f"Removed file: {path_obj}")
            except OSError as exc:
                logger.warning(f"Failed to remove {path_obj}: {exc}")
