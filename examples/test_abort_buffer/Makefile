#=======================================================================
#                   define the compiler names
#=======================================================================
include ../make.inc

PY_MOD      = pywrapper
WRAPFLAGS   = -v
F2PY        = f2py-f90wrap
F90_SRC     = main_A.f90 main_B.f90
F90WRAP_SRC = $(addprefix f90wrap_,${F90_SRC})
OBJ         = $(F90_SRC:.f90=.o)
F90WRAP_OBJ = $(F90WRAP_SRC:.f90=.o)
SRC_LIB 		= src
F90WRAP_LIB  = f90wrap
.PHONY: all clean

all: test

clean:
	rm -rf *.mod *.smod *.o lib*.a lib*.so f90wrap*.f90 ${PY_MOD}.py _${PY_MOD}*.so __pycache__/ .f2py_f2cmap build ${PY_MOD}* signatures_*

# This can be done in parallel as previously
%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

# This can be done in parallel as previously
f90wrap_%.f90: %.o
	${F90WRAP} -m ${PY_MOD}_$(patsubst %.o,%,$<) ${WRAPFLAGS} $(patsubst %.o,%.f90,$<)

# Using 2 f2py-f90wrap steps enables to compile f90wrap_*.f90 separately rather than compiling them through f2py-f90wrap invocation. Compiling them separatly enables to compile them in parallel.
# This can also be done in parallel but probably not critical
signatures_%.pyf: f90wrap_%.f90
	${F2PY} $< -m _${PY_MOD}_$(patsubst f90wrap_%.f90,%,$<) -h $@

# This only compile c interfaces no fortran interfaces
# This can now be done in parallel
f2py_%: signatures_%.pyf lib${SRC_LIB}.so lib${F90WRAP_LIB}.so
	${F2PY} ${F2PYFLAGS} -c $< -L. -l${SRC_LIB} -l${F90WRAP_LIB}

# Compile f90wrap generated interfaces
# This can now be done in parallel
# This also ease support for fortran compiler other than gfortran
f90wrap_%.o: f90wrap_%.f90
	${F90} ${F90FLAGS} -c $< -o $@

lib${SRC_LIB}.so: ${OBJ}
	${F90} ${F90FLAGS} -shared -o lib${SRC_LIB}.so ${OBJ} ${LDFLAGS}

lib${F90WRAP_LIB}.so: ${F90WRAP_OBJ}
	${F90} ${F90FLAGS} -shared -o lib${F90WRAP_LIB}.so ${F90WRAP_OBJ} ${LDFLAGS}

test: f2py_main_A f2py_main_B
	LD_LIBRARY_PATH=. ${PYTHON} tests.py
