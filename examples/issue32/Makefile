include ../make.inc
F90FLAGS += -x f95-cpp-input

all: test

test.o: test.f90
	$(F90) $(F90FLAGS) -c $<

mod: test.o
	f90wrap -m test ./test.f90
	f2py-f90wrap $(F2PYFLAGS) -c -m _test f90wrap_toplevel.f90 test.o

test: mod
	$(PYTHON) -c 'import test; test.foo(1.0, 2)'

clean:
	-rm -f test.py
	-rm -f f90wrap*.f90 *.o *.pyc *.mod
	-rm -f _test*.so _testpkg*.so
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
