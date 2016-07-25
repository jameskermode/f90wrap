FPP=gfortran
FFLAGS=-fPIC

%.o : %.f90
	${FPP} ${FFLAGS} -c $< -o $@

all: test.o
	f90wrap -m test -P test.f90
	f2py-f90wrap -c -m _test f90wrap_test.f90 test.o

test: all
	python test.py

clean:
	-rm *.o f90wrap*.f90 *.so *.mod
	-rm -r test/

