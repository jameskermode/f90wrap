FC = gfortran
#FPP = ifort
FCFLAGS = -fPIC
PYTHON = python

%.o : %.f
	${FC} ${FCFLAGS} -c $< -o $@

all: subroutine_oldstyle.o
	f90wrap -m itest -P subroutine_oldstyle.f -v
	f2py-f90wrap --fcompiler=$(FC) --build-dir . -c -m _itest f90wrap_toplevel.f90 \
		subroutine_oldstyle.o

test: all
	$(PYTHON) run.py

clean:
	rm -f *.o f90wrap*.f *.so *.mod
	rm -rf src.*/
	rm -rf itest/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/
