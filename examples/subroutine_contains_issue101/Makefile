F = gfortran 
CFLAGS = -fPIC

all: test

%.o : %.f90
	${FC} ${CFLAGS} -c $< -o $@

test.o: test.f90

test.py: test.o
	f90wrap -m test test.f90
	f2py-f90wrap -c -m _test f90wrap_toplevel.f90 test.o

test: test.py
	python run.py

clean:
	-rm -f *.o f90wrap*.f90 *.so *.mod *.pyc
	-rm -rf __pycache__
	-rm -f test.py
