include ../make.inc
F90FLAGS += -x f95-cpp-input

SRC_AR = libsrc.a

all: test

test.o: test.f90
	$(F90) $(F90FLAGS) -c $<

mod: test.o
	f90wrap -m test ./test.f90
	ar rcs ${SRC_AR} test.o
	f2py-f90wrap $(F2PYFLAGS) -c -m _test f90wrap_toplevel.f90 -L. -lsrc

test: mod
	$(PYTHON) -c 'import test; test.foo(1.0, 2)'

clean:
	-rm -f test.py
	-rm -f f90wrap*.f90 *.o ${SRC_AR} *.pyc *.mod
	-rm -f _test*.so _testpkg*.so
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
