#=======================================================================
#                   define the compiler names
#=======================================================================

include ../make.inc
PY_MOD      = pywrapper
F90_SRC     = main.f90 array_type.f90
OBJ         = $(F90_SRC:.f90=.o)
WRAPFLAGS   = -v --type-check
F2PY        = f2py-f90wrap
JSON_FILE   = dump_package.json
.PHONY: all clean

all: test

clean:
	-rm -rf *.mod *.smod *.o *.so f90wrap*.f90 ${PY_MOD}_* __pycache__/ .f2py_f2cmap build ${JSON_FILE}

main.o: main.f90 array_type.o
	${F90} ${F90FLAGS} -c $< -o $@

%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

f90wrap_array_type.f90: array_type.f90
	${F90WRAP} --dump-package ${JSON_FILE} -m ${PY_MOD}_$(<:.f90=) ${WRAPFLAGS} $<

f90wrap_main.f90: main.f90 ${JSON_FILE}
	${F90WRAP} --external-packages ${JSON_FILE} -m ${PY_MOD}_$(<:.f90=) ${WRAPFLAGS} $<

${JSON_FILE}: f90wrap_array_type.f90

f2py_%: f90wrap_%.f90 ${OBJ}
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD}_$(<:f90wrap_%.f90=%) ${F2PYFLAGS} $< ${OBJ}

test: f2py_main f2py_array_type
	${PYTHON} tests.py
