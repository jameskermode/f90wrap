include ../make.inc

SRC_AR = libsrc.a

%.o : %.f90
	${F90} ${F90FLAGS} -c $< -I. -o $@

all: main.o
	$(F90WRAP) -m itest -P main.f90 -v
	ar rcs ${SRC_AR} main.o
	f2py-f90wrap $(F2PYFLAGS) -c -m _itest f90wrap_main.f90 -L. -lsrc

test: all
	$(PYTHON) run.py

clean:
	rm -f *.o ${SRC_AR} f90wrap*.f90 *.so *.mod
	rm -rf src.*/
	rm -rf itest/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
