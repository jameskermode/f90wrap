f2py = [py3, '-m', 'numpy.f2py', '@INPUT@', '--build-dir', '@OUTDIR@', '--lower']

sizeof_source = custom_target(
    input: 'sizeoffortran.f90',
    output: 'sizeof_fortran_tmodule.c',
    command: f2py + ['-m', 'sizeof_fortran_t']
)

py3.extension_module(
    'sizeof_fortran_t',
    'sizeoffortran.f90', sizeof_source,
    dependencies: fortranobject_dep,
    install: true,
    subdir: 'f90wrap',
)

py3.extension_module(
    'arraydata',
    'arraydatamodule.c',
    dependencies: fortranobject_dep,
    install: true,
    subdir: 'f90wrap',
)

py3.install_sources(
    'codegen.py',
    'f90wrapgen.py',
    'fortran.py',
    'fortrantype.py',
    '__init__.py',
    'latex.py',
    '__main__.py',
    'parser.py',
    'pywrapgen.py',
    'runtime.py',
    'six.py',
    'transform.py',
    'scripts/f2py_f90wrap.py',
    'scripts/f90doc.py',
    'scripts/__init__.py',
    'scripts/main.py',
    subdir: 'f90wrap',
    preserve_path: true,
)
