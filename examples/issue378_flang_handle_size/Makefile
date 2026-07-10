include ../make.inc

PY_MOD = example
F90_SRC = example.f90
F90WRAP_SRC = f90wrap_example.f90
F90WRAP_FLAGS = --direct-c --fortran-compiler "$(F90) $(F90FLAGS)" -m $(PY_MOD)

.PHONY: all clean test

all: test

clean:
	rm -f *.mod *.o f90wrap_*.f90 _*.c $(PY_MOD).py .f2py_f2cmap

test: clean
	PYTHONPATH=../.. $(PYTHON) -m f90wrap $(F90WRAP_FLAGS) $(F90_SRC)
	$(F90) $(F90FLAGS) -c $(F90_SRC) -o example.o
	$(F90) $(F90FLAGS) -c $(F90WRAP_SRC) -o f90wrap_example.o
