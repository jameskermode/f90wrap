#=======================================================================
#                   define the compiler names
#=======================================================================

F90         = gfortran
PYTHON      = python3
CFLAGS      = -fPIC
F90FLAGS    = -fPIC
PY_MOD      = pywrapper
F90_SRC     = main.f90
OBJ         = $(F90_SRC:.f90=.o)
SIGNATURES	= _signatures.pyf
F2PYFLAGS   = --build-dir build
F2PY        = f2py
LINK		= -lgfortran
.PHONY: all clean

all: test

clean:
	rm -rf *.mod *.smod *.o f90wrap*.f90 ${PY_MOD}_*.py _${PY_MOD}*.so __pycache__/ .f2py_f2cmap build ${PY_MOD}_*/ ${SIGNATURES}

%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

${SIGNATURES}: ${F90_SRC}
	${F2PY} ${F90_SRC} -m _${PY_MOD}_sign -h ${SIGNATURES}

f2py: ${OBJ} ${SIGNATURES}
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD}_sign ${F2PYFLAGS} ${LINK} ${OBJ} ${SIGNATURES}
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD}_no_sign ${F2PYFLAGS} ${F90_SRC}

test: f2py
	${PYTHON} tests_sign.py
	${PYTHON} tests_no_sign.py
