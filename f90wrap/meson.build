f2py = [py3, '-m', 'numpy.f2py', '@INPUT@', '--build-dir', '@OUTDIR@', '--lower']

sizeof_source = custom_target(
    input: 'sizeoffortran.f90',
    output: 'sizeof_fortran_tmodule.c',
    command: f2py + ['-m', 'sizeof_fortran_t']
)

# Only build Fortran extensions if Fortran compiler is available
# Fortran language was optionally added in root meson.build
link_args = []
# Try to add fortran again (will succeed if already added, or if available)
if add_languages('fortran', native: false, required: false)
    fortran_compiler = meson.get_compiler('fortran')
    if fortran_compiler.get_id() == 'nvidia_hpc'
        link_args = ['-lnvf', '-lrt']
    endif

    py3.extension_module(
        'sizeof_fortran_t',
        'sizeoffortran.f90', sizeof_source,
        dependencies: fortranobject_dep,
        install: true,
        subdir: 'f90wrap',
        link_args: link_args,
    )
endif

py3.extension_module(
    'arraydata',
    'arraydatamodule.c',
    dependencies: fortranobject_dep,
    install: true,
    subdir: 'f90wrap',
    link_args: link_args,
)

py3.install_sources(
    'build.py',
    'codegen.py',
    'directc.py',
    'setuptools_ext.py',
    'directc_cgen/__init__.py',
    'directc_cgen/arguments.py',
    'directc_cgen/arguments_array.py',
    'directc_cgen/arguments_scalar.py',
    'directc_cgen/derived_types.py',
    'directc_cgen/module_helpers_array.py',
    'directc_cgen/module_helpers.py',
    'directc_cgen/procedures.py',
    'directc_cgen/procedures_return.py',
    'directc_cgen/utils.py',
    'f90wrapgen.py',
    'fortran.py',
    'fortrantype.py',
    '__init__.py',
    'latex.py',
    '__main__.py',
    'numpy_utils.py',
    'parser.py',
    'pywrapgen.py',
    'runtime.py',
    'transform.py',
    'scripts/f2py_f90wrap.py',
    'scripts/f90doc.py',
    'scripts/__init__.py',
    'scripts/main.py',
    subdir: 'f90wrap',
    preserve_path: true,
)
