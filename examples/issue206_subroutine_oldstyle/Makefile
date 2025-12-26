include ../make.inc
SRC_AR = libsrc.a
%.o : %.f
	${F90} ${F90FLAGS} -c $< -o $@

all: subroutine_oldstyle.o
	f90wrap -m itest -P subroutine_oldstyle.f -v
	ar rcs ${SRC_AR} subroutine_oldstyle.o
	f2py-f90wrap $(F2PYFLAGS) -c -m _itest f90wrap_toplevel.f90 -L. -lsrc

test: all
	$(PYTHON) run.py

clean:
	rm -f *.o ${SRC_AR} f90wrap*.f f90wrap*.f90 *.so *.mod
	rm -rf src.*/
	rm -rf itest/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
