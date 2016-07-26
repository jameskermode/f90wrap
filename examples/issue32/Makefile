CC = gcc
F90 = gfortran
FPP = gfortran -E
F90FLAGS = -x f95-cpp-input -fPIC
CFLAGS = -fPIC

all: test

test.o: test.f90
	$(F90) $(F90FLAGS) -c $<

mod: test.o
	f90wrap -m test ./test.f90
	f2py-f90wrap -c -m _test f90wrap_toplevel.f90 test.o

test: mod
	python -c 'import test; test.foo(1.0, 2)'

clean:
	-rm test.py
	-rm f90wrap*.f90 *.o *.pyc *.mod
	-rm _test.so _testpkg.so

