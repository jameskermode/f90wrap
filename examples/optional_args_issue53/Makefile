include ../make.inc

SRC_AR = libsrc.a

%.o : %.f90
	${F90} ${F90FLAGS} -c $< -o $@

all: main.o
	f90wrap -m test -P main.f90
	ar rcs ${SRC_AR} main.o
	f2py-f90wrap $(F2PYFLAGS) -c -m _test f90wrap_toplevel.f90 -L. -lsrc

test: all
	$(PYTHON) run.py

clean:
	-rm -f *.o ${SRC_AR} f90wrap*.f90 *.so *.mod
	-rm -rf test/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
