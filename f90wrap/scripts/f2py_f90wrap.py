#!/usr/bin/env python

#  f90wrap: F90 to Python interface generator with derived type support
#
#  Copyright James Kermode 2011-2018
#
#  This file is part of f90wrap
#  For the latest version see github.com/jameskermode/f90wrap
#
#  f90wrap is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  f90wrap is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with f90wrap. If not, see <http://www.gnu.org/licenses/>.
#
#  If you would like to license the source code under different terms,
#  please contact James Kermode, james.kermode@gmail.com

"""
This script patches :mod:`numpy.f2py` at runtime, to customise the C code that
is generated. We make several changes to f2py:

  1. Allow the Fortran :c:func:`present` function to work correctly with optional arguments.
     If an argument to an f2py wrapped function is optional and is not given, replace it
     with ``NULL``.

  2. Allow Fortran routines to raise a :exc:`RuntimeError` exception
     with a message by calling an external function
     :c:func:`f90wrap_abort`. This is implemented using a
     :c:func:`setjmp`/ :c:func:`longjmp` trap.

  3. Allow Fortran routines to be interrupted with :kbd:`Ctrl+C` by installing a custom
     interrupt handler before the call into Fortran is made. After the Fortran routine
     returns, the previous interrupt handler is restored.

"""

from __future__ import print_function
import sys
from numpy.f2py.auxfuncs import *

# __all__ = []

def main():

    import numpy
    numpy_version = tuple([int(x) for x in numpy.__version__.split('.')[0:2]])
    if not numpy_version >= (1,13):
       raise ImportError('f2py-f90wrap tested with numpy version 1.13 or later, found version %s' % numpy.__version__)


    import numpy.f2py.auxfuncs

    import numpy.f2py.capi_maps
    numpy.f2py.capi_maps.cformat_map['long_long'] = '%Ld'

    import numpy.f2py.rules, numpy.f2py.cb_rules

    includes_inject = "#includes0#\n"

    includes_inject = includes_inject + """

    /* custom abort handler - James Kermode <james.kermode@gmail.com> */
    """
    if(sys.platform == 'win32'):
       includes_inject = includes_inject + "#include <signal.h> // fix https://github.com/jameskermode/f90wrap/issues/73 \n#include <setjmpex.h> // fix https://github.com/jameskermode/f90wrap/issues/96\n"
    else:
       includes_inject = includes_inject + "#include <signal.h>\n#include <setjmp.h>\n"

    includes_inject = includes_inject + """
    #define ABORT_BUFFER_SIZE 1024
    extern jmp_buf environment_buffer;
    extern char abort_message[ABORT_BUFFER_SIZE];
    void f90wrap_abort_(char *message, int len);
    void f90wrap_abort_int_handler(int signum);

    #include <stdlib.h>
    #include <string.h>

    jmp_buf environment_buffer;
    char abort_message[ABORT_BUFFER_SIZE];

    void f90wrap_abort_(char *message, int len_message)
    {
      strncpy(abort_message, message, ABORT_BUFFER_SIZE);
      abort_message[ABORT_BUFFER_SIZE-1] = '\\0';
      longjmp(environment_buffer, 0);
    }

    // copy of f90wrap_abort_ with a second underscore
    // void (*f90wrap_abort__)(char *, int) = &f90wrap_abort_;
    void f90wrap_abort__(char *message, int len_message)
    {
      strncpy(abort_message, message, ABORT_BUFFER_SIZE);
      abort_message[ABORT_BUFFER_SIZE-1] = '\\0';
      longjmp(environment_buffer, 0);
    }


    void f90wrap_abort_int_handler(int signum)
    {
      char message[] = "Interrupt occured";
      f90wrap_abort_(message, strlen(message));
    }

    /* end of custom abort handler  */
    """

    numpy.f2py.rules.module_rules['modulebody'] = numpy.f2py.rules.module_rules['modulebody'].replace("#includes0#\n", includes_inject)

    numpy.f2py.rules.routine_rules['body'] = numpy.f2py.rules.routine_rules['body'].replace("volatile int f2py_success = 1;\n", """volatile int f2py_success = 1;
        int setjmpvalue; /* James Kermode - for setjmp */
    """)

    numpy.f2py.rules.routine_rules['body'] = numpy.f2py.rules.routine_rules['body'].replace('#callfortranroutine#\n', """/* setjmp() exception handling added by James Kermode */
    PyOS_sighandler_t _npy_sig_save;
    _npy_sig_save = PyOS_setsig(SIGINT, f90wrap_abort_int_handler);
    setjmpvalue = setjmp(environment_buffer);
    if (setjmpvalue != 0) {
      PyOS_setsig(SIGINT, _npy_sig_save);
      PyObject *err_msg = PyUnicode_DecodeUTF8(abort_message, strlen(abort_message), "replace");
      if (err_msg != NULL) {
        PyErr_SetObject(PyExc_RuntimeError, err_msg);
        Py_DECREF(err_msg);
      } else {
        PyErr_SetString(PyExc_RuntimeError, "Error decoding abort message");
      }
    } else {
     #callfortranroutine#
     PyOS_setsig(SIGINT, _npy_sig_save);
    }
    /* End addition */
    """)


    numpy.f2py.auxfuncs.options['persistant_callbacks'] = True

    # Disable callback argument cleanup so that callbacks can be called after function returns.
    # This will lead to a small memory leak every time a function with callback arguments is called.
    def persistant_callbacks(var):
       return True

    numpy.f2py.cb_rules.cb_routine_rules['body'] = numpy.f2py.cb_rules.cb_routine_rules['body'].replace('capi_longjmp_ok = 1', 'capi_longjmp_ok = 0')

    # Fix for issue #204: When callbacks are called indirectly from Fortran (not through
    # a Python wrapper), the callback context is not set up, so cb->nofargs is 0.
    # This causes arguments to not be passed to the callback. We fix this by:
    # 1. Always setting nofargs to maxnofargs (whether using cb_local or persistent context)
    # 2. Always creating a fresh tuple for the arguments to avoid stale state from previous calls

    # Set nofargs when falling back to cb_local
    numpy.f2py.cb_rules.cb_routine_rules['body'] = numpy.f2py.cb_rules.cb_routine_rules['body'].replace(
        'cb = &cb_local;\n    }',
        'cb = &cb_local;\n        cb->nofargs = #maxnofargs#;\n    }'
    )

    # Always ensure nofargs is set correctly and create a fresh tuple for each invocation.
    # This handles both the cb_local case and the persistent callback context case.
    # The persistent context may have stale args_capi from a previous call.
    numpy.f2py.cb_rules.cb_routine_rules['body'] = numpy.f2py.cb_rules.cb_routine_rules['body'].replace(
        'capi_arglist = cb->args_capi;\n',
        'capi_arglist = NULL; /* Always create fresh tuple for indirect calls */\n    cb->nofargs = #maxnofargs#;\n'
    )

    # Create a properly sized tuple (was empty before)
    numpy.f2py.cb_rules.cb_routine_rules['body'] = numpy.f2py.cb_rules.cb_routine_rules['body'].replace(
        'capi_arglist = (PyTupleObject *)Py_BuildValue("()");',
        'capi_arglist = (PyTupleObject *)PyTuple_New(#maxnofargs#);'
    )

    numpy.f2py.rules.arg_rules[7]['cleanupfrompyobj'] = {l_not(persistant_callbacks): numpy.f2py.rules.arg_rules[7]['cleanupfrompyobj'],
                                                         persistant_callbacks: '}'}

    numpy.f2py.rules.arg_rules[8]['callfortran'] = {isintent_c:'#varname#,',
                                                    l_and(isoptional,l_not(isintent_c)):'#varname#_capi == Py_None ? NULL : &#varname#,',
                                                    l_and(l_not(isoptional),l_not(isintent_c)):'&#varname#,'}


    numpy.f2py.rules.arg_rules[14]['callfortran'] = {isintent_c:'#varname#,',l_and(isoptional,l_not(isintent_c)):'#varname#_capi == Py_None ? NULL : &#varname#,',
                                                     l_and(l_not(isoptional),l_not(isintent_c)):'&#varname#,'}

    numpy.f2py.rules.arg_rules[21]['callfortran'] = {isintent_out:'#varname#,', l_and(isoptional, l_not(isintent_out)):'#varname#_capi == Py_None ? NULL : #varname#,',
                                                     l_and(l_not(isoptional), l_not(isintent_out)): '#varname#,'}


    numpy.f2py.rules.arg_rules[26]['callfortran'] = {isintent_out:'#varname#,', l_and(isoptional, l_not(isintent_out)):'#varname#_capi == Py_None ? NULL : #varname#,',
                                                     l_and(l_not(isoptional), l_not(isintent_out)): '#varname#,'}

    if numpy_version < (1,24):
        numpy.f2py.rules.arg_rules[33]['frompyobj'].insert(2, {isoptional: 'if (#varname#_capi != Py_None) {'})
        numpy.f2py.rules.arg_rules[33]['frompyobj'].insert(5, {isoptional: '}'})

        del numpy.f2py.rules.arg_rules[33]['frompyobj'][6]
        numpy.f2py.rules.arg_rules[33]['frompyobj'].insert(6, {l_not(isoptional): \
        "\n\t\tif (capi_#varname#_tmp == NULL) {"
        "\n\t\t\tif (!PyErr_Occurred())"
        "\n\t\t\t\tPyErr_SetString(#modulename#_error,\"failed in converting #nth# `#varname#\' of #pyname# to C/Fortran array\" );"
        "\n\t\t} else {"
        "\n\t\t\t#varname# = (#ctype# *)(capi_#varname#_tmp->data);"
        })

        numpy.f2py.rules.arg_rules[33]['frompyobj'].insert(7, {isoptional:\
        "\n\t\tif (#varname#_capi != Py_None && capi_#varname#_tmp == NULL) {"
        "\n\t\t\tif (!PyErr_Occurred())"
        "\n\t\t\t\tPyErr_SetString(#modulename#_error,\"failed in converting #nth# `#varname#\' of #pyname# to C/Fortran array\" );"
        "\n\t\t} else {"
        "\n\t\t\tif (#varname#_capi != Py_None) #varname# = (#ctype# *)(capi_#varname#_tmp->data);"
        })

    else:
        numpy.f2py.rules.arg_rules[33]['frompyobj'].insert(3, {isoptional: '\tif (#varname#_capi != Py_None) {'})
        numpy.f2py.rules.arg_rules[33]['frompyobj'].insert(6, {isoptional: '\t}'})

        del numpy.f2py.rules.arg_rules[33]['frompyobj'][7]
        numpy.f2py.rules.arg_rules[33]['frompyobj'].insert(7, {l_not(isoptional):\
        "\n\t\tif (capi_#varname#_as_array == NULL) {"
        "\n\t\t\tPyObject* capi_err = PyErr_Occurred();"
        "\n\t\t\tif (capi_err == NULL) {"
        "\n\t\t\t\tcapi_err = #modulename#_error;"
        "\n\t\t\t\tPyErr_SetString(capi_err, capi_errmess);"
        "\n\t\t\t}"
        "\n\t\t} else {"
        "\n\t\t\t#varname# = (#ctype# *)(PyArray_DATA(capi_#varname#_as_array));"
        })

        del numpy.f2py.rules.arg_rules[33]['frompyobj'][8]
        numpy.f2py.rules.arg_rules[33]['frompyobj'].insert(8, {l_and(isstringarray, isoptional):\
        "\t\tif (#varname#_capi != Py_None) slen(#varname#) = f2py_itemsize(#varname#);"
        })
        numpy.f2py.rules.arg_rules[33]['frompyobj'].insert(9, {l_and(isstringarray, l_not(isoptional)):\
        "\t\tslen(#varname#) = f2py_itemsize(#varname#);"
        })

        numpy.f2py.rules.arg_rules[33]['frompyobj'].insert(10, {isoptional:\
        "\n\t\tif (#varname#_capi != Py_None && capi_#varname#_as_array == NULL) {"
        "\n\t\t\tPyObject* capi_err = PyErr_Occurred();"
        "\n\t\t\tif (capi_err == NULL) {"
        "\n\t\t\t\tcapi_err = #modulename#_error;"
        "\n\t\t\t\tPyErr_SetString(capi_err, capi_errmess);"
        "\n\t\t\t}"
        "\n\t\t} else {"
        "\n\t\t\tif (#varname#_capi != Py_None) #varname# = (#ctype# *)(PyArray_DATA(capi_#varname#_as_array));"
        })

    # now call the main function
    print('\n!! f90wrap patched version of f2py - James Kermode <james.kermode@gmail.com> !!\n')

    # Force meson backend for NumPy >= 2.0 (distutils was removed)
    if numpy_version >= (2, 0):
        if '--backend' not in sys.argv:
            print('\nNumPy 2.0+ detected, using meson backend (distutils was removed).')
            sys.argv.insert(1, '--backend')
            sys.argv.insert(2, 'meson')

    # Monkey-patch numpy's meson backend to fix include and library paths
    # for separate build directories when using --build-dir
    import os
    from pathlib import Path

    build_dir_to_patch = None
    if '--build-dir' in sys.argv:
        build_dir_idx = sys.argv.index('--build-dir') + 1
        if build_dir_idx < len(sys.argv):
            build_dir = sys.argv[build_dir_idx]
            # Only patch if build_dir is not '.' (separate directory)
            if build_dir != '.':
                build_dir_to_patch = build_dir

    if build_dir_to_patch:
        # Monkey-patch the meson backend's write_meson_build method
        try:
            from numpy.f2py._backends import _meson
            original_write_meson_build = _meson.MesonBackend.write_meson_build

            def patched_write_meson_build(self, build_dir):
                # Call original method to generate meson.build
                original_write_meson_build(self, build_dir)

                # Now patch the generated file
                meson_build = Path(build_dir) / 'meson.build'
                if meson_build.exists():
                    content = meson_build.read_text()
                    modified = False

                    # Add include path for parent directory (for .mod files)
                    if 'inc_parent = include_directories' not in content:
                        content = content.replace(
                            "inc_np = include_directories(incdir_numpy, incdir_f2py)",
                            "inc_np = include_directories(incdir_numpy, incdir_f2py)\ninc_parent = include_directories('..')"
                        )
                        modified = True

                        # Also add inc_parent to the include_directories list in py.extension_module
                        # Look for the include_directories list and add inc_parent if not already there
                        import re
                        # Find the include_directories section in extension_module
                        pattern = r'(include_directories:\s*\[\s*inc_np,)'
                        if re.search(pattern, content):
                            content = re.sub(pattern, r'\1\n                     inc_parent,', content)
                            modified = True

                    # Replace '''.''' with inc_parent in include_directories list (for -I. flag)
                    if "'''.'''," in content:
                        content = content.replace("'''.''',", "inc_parent,")
                        modified = True

                    # Fix library search path to point to parent directory
                    if "lib_dir_0 = declare_dependency(link_args : ['''-L.'''])" in content:
                        content = content.replace(
                            "lib_dir_0 = declare_dependency(link_args : ['''-L.'''])",
                            "lib_dir_0 = declare_dependency(link_args : ['''-L../..'''])"
                        )
                        modified = True

                    # Add Fortran source files corresponding to .o files
                    # Meson doesn't handle .o files properly, need to compile from source
                    # Collect .o files from command line
                    fortran_obj_files = []
                    for arg in sys.argv:
                        if arg.endswith('.o'):
                            fortran_obj_files.append(arg)

                    if fortran_obj_files:
                        import re
                        # Convert .o files to .f90 files and add them to py.extension_module sources
                        additional_sources = []
                        for obj_file in fortran_obj_files:
                            # Try .f90, .F90, and .f extensions
                            for ext in ['.f90', '.F90', '.f']:
                                f90_file = obj_file.replace('.o', ext)
                                if os.path.exists(f90_file):
                                    # Get relative path from build directory to source
                                    rel_path = os.path.join('..', os.path.basename(f90_file))
                                    additional_sources.append(f"                     '''{rel_path}''',")
                                    break

                        if additional_sources:
                            # Find the sources list in py.extension_module and add our files
                            # Look for the pattern: py.extension_module('name', [ ... fortranobject_c ], ...)
                            # We want to insert before fortranobject_c
                            pattern = r'(py\.extension_module\([^,]+,\s*\[[^\]]*)(fortranobject_c)'
                            match = re.search(pattern, content, re.DOTALL)
                            if match:
                                # Insert additional sources before fortranobject_c
                                new_content = match.group(1) + '\n'.join(additional_sources) + '\n                     ' + match.group(2)
                                content = content[:match.start()] + new_content + content[match.end():]
                                modified = True

                    if modified:
                        meson_build.write_text(content)
                        print(f"\nPatched {meson_build} for separate build directory")

            _meson.MesonBackend.write_meson_build = patched_write_meson_build
        except (ImportError, AttributeError):
            # numpy doesn't have meson backend (older version), no patching needed
            pass

    numpy.f2py.main()


if __name__ == "__main__":
    sys.exit(main())
