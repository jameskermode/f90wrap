# Makefile for name collision test example

F90 = gfortran
F2PY = f2py-f90wrap
F90WRAP = f90wrap

F90FLAGS = -fPIC
F2PYFLAGS = --f90flags="$(F90FLAGS)"

.PHONY: all test clean

all: test_collision

test_collision: test_module.f90
	# Generate wrapper with f90wrap
	$(F90WRAP) -m test_collision test_module.f90
	# Compile with f2py
	$(F2PY) -c $(F2PYFLAGS) -m _test_collision f90wrap_*.f90 test_module.f90

test: test_collision
	python3 test_name_collision.py

clean:
	rm -f *.o *.mod *.so f90wrap_*.f90 test_collision.py
	rm -rf test_collision __pycache__
