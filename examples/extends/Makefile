FPP=gfortran
#FPP=ifort
FFLAGS=-fPIC

%.o : %.f90
	${FPP} ${FFLAGS} -c $< -o $@

all: testextends.o
	f90wrap -m testextends testextends.f90
	f2py-f90wrap -c -m _testextends f90wrap_*.f90

test: all
	python testextends.py

clean:
	-rm *.o f90wrap*.f90 *.so *.mod

