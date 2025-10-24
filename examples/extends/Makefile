include ../make.inc

%.o : %.f90
	${F90} ${F90FLAGS} -c $< -o $@

all: test

testextends: testextends.o
	f90wrap -m testextends testextends.f90
	f2py-f90wrap $(F2PYFLAGS) -c -m _testextends f90wrap_*.f90 *.o

test: testextends
	$(PYTHON) testextends.py

clean:
	-rm -f *.o f90wrap*.f90 *.so *.mod testextends.py
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
