include ../make.inc

%.o : %.f90
	${F90} ${F90FLAGS} -c $< -o $@

all: test.o
	f90wrap -m test -P test.f90
	f2py-f90wrap $(F2PYFLAGS) -c -m _test f90wrap_test.f90 test.o

clean:
	-rm -f *.o f90wrap*.f90 *.so *.mod
	-rm -rf test/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/

test: all
	$(PYTHON) -c 'import test'
