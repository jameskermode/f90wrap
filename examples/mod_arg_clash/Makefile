include ../make.inc

SRC_AR = libsrc.a

%.o : %.f90
	${F90} ${F90FLAGS} -c $< -o $@

all: test.o
	f90wrap -m test -P test.f90
	ar rcs ${SRC_AR} test.o
	f2py-f90wrap $(F2PYFLAGS) -c -m _test f90wrap_test.f90 -L. -lsrc

test: all
	$(PYTHON) tests.py

clean:
	-rm -f *.o ${SRC_AR} f90wrap*.f90 *.so *.mod
	-rm -rf test/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
