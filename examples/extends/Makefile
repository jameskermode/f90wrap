include ../make.inc

SRC_AR = libsrc.a

%.o : %.f90
	${F90} ${F90FLAGS} -c $< -o $@

all: test

testextends: testextends.o
	f90wrap -m testextends testextends.f90
	ar rcs ${SRC_AR} *.o
	f2py-f90wrap $(F2PYFLAGS) -c -m _testextends f90wrap_*.f90 -L. -lsrc

test: testextends
	$(PYTHON) testextends.py

clean:
	-rm -f *.o ${SRC_AR} f90wrap*.f90 *.so *.mod testextends.py
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
