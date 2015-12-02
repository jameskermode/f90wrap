CC = gcc
F90 = gfortran
FPP = gfortran -E
F90FLAGS = -x f95-cpp-input -fPIC
CFLAGS = -fPIC

all: test

example.o: example.f90
	$(F90) $(F90FLAGS) -c $<

mod: example.o
	f90wrap -m example ./example.f90
	f2py-f90wrap -c -m _example f90wrap_example.f90 example.o

package: example.o
	f90wrap -P -m examplepkg ./example.f90
	f2py-f90wrap -c -m _examplepkg f90wrap_example.f90 example.o

test: mod package
	python test.py

clean:
	-rm -r examplepkg
	-rm example.py
	-rm f90wrap*.f90 *.o *.pyc *.mod
	-rm _example.so _examplepkg.so

