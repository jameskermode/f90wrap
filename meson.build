# much of this file is derived from SciPy

project('f90wrap',
    'c',
    version: run_command('get_version.py', check: true).stdout().strip(),
    meson_version: '>= 0.64.0',
)

# force rebuild to re-calculate the version if it changes
import('fs').read('f90wrap/__init__.py')

# Adding at project level causes many spurious -lgfortran flags.
add_languages('fortran', native: false)

py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()

incdir_numpy = run_command(
    py3, '-c',
    'import numpy; print(numpy.get_include())',
    check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

# Share this object across multiple modules.
fortranobject_lib = static_library('_fortranobject',
  fortranobject_c,
  dependencies: py3_dep,
  include_directories: [inc_np, inc_f2py],
)
fortranobject_dep = declare_dependency(
  link_with: fortranobject_lib,
  include_directories: [inc_np, inc_f2py],
)

subdir('f90wrap')
