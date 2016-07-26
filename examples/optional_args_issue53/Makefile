FPP=gfortran
#FPP=ifort
FFLAGS=-fPIC

%.o : %.f90
	${FPP} ${FFLAGS} -c $< -o $@

all: main.o
	f90wrap -m test -P main.f90
	f2py-f90wrap -c -m _test f90wrap_toplevel.f90 main.o

test: all
	python run.py

clean:
	-rm *.o f90wrap*.f90 *.so *.mod
	-rm -r test/

