#=======================================================================
#                   define the compiler names
#=======================================================================

include ../make.inc
PY_MOD      = pywrapper
F90_SRC     = $(filter-out f90wrap_%.f90,$(wildcard *.f90))
OBJ         = $(F90_SRC:.f90=.o)
F90WRAP_SRC = $(addprefix f90wrap_,${F90_SRC})
WRAPFLAGS   = -v --package --relative
F2PY        = f2py-f90wrap
TOP_MODULE  = top_module
.PHONY: all clean f90wrap f2py test topmodule

all: test

clean:
	rm -rf *.mod *.smod *.o f90wrap*.f90 ${PY_MOD}.py _${PY_MOD}*.so __pycache__/ .f2py_f2cmap build ${PY_MOD}/ ${TOP_MODULE}

# Dependencies between Fortran modules
base_type.o:
inheritance_type.o: base_type.o
composition_type.o: base_type.o
main.o: $(filter-out main.o, ${OBJ})

%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

f90wrap: ${OBJ}
	${F90WRAP} -m ${PY_MOD} ${WRAPFLAGS} ${F90_SRC}

f2py: f90wrap
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD} ${F2PYFLAGS} f90wrap_*.f90 *.o

test: topmodule
	${PYTHON} tests.py

topmodule: f2py
	rm -rf ${TOP_MODULE}
	mkdir ${TOP_MODULE}
	mv ${PY_MOD}/ _${PY_MOD}.*so ${TOP_MODULE}/
	touch ${TOP_MODULE}/__init__.py
