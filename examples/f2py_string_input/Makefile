#=======================================================================
#                   define the compiler names
#=======================================================================

include ../make.inc
PYTHON      = python3
PY_MOD      = pywrapper
F90_SRC     = main.f90
OBJ         = $(F90_SRC:.f90=.o)
SIGNATURES	= _signatures.pyf
F2PY        = f2py
ifeq ($(notdir $(F90)),gfortran)
	LINK				= -lgfortran
endif
.PHONY: all clean

all: f2py-ko test

clean:
	rm -rf *.mod *.smod *.o f90wrap*.f90 ${PY_MOD}.py _${PY_MOD}*.so __pycache__/ .f2py_f2cmap build ${PY_MOD}/ ${SIGNATURES}

%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

${SIGNATURES}: ${F90_SRC}
	${F2PY} ${F90_SRC} -m _${PY_MOD} -h ${SIGNATURES}

f2py-ko: ${OBJ} ${SIGNATURES}
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD} ${F2PYFLAGS} ${LINK} ${OBJ} ${SIGNATURES}

f2py-ok: ${F90_SRC}
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD} ${F2PYFLAGS} ${F90_SRC}

test: f2py-ok
	${PYTHON} test.py
