include ../make.inc

SRC_AR = libsrc.a

test :  clean functions.o wrap python
	$(PYTHON) tests.py
functions.o : functions.f90
	$(F90) $(F90FLAGS) -c -O3 functions.f90
wrap :
	f90wrap --default-to-inout -m test_python functions.f90
python :
	ar rcs ${SRC_AR} *.o
	f2py-f90wrap $(F2PYFLAGS) -c -m _test_python f90wrap_functions.f90 -L. -lsrc
clean :
	rm -f *.o ${SRC_AR} *.mod *.so f90wrap* test_python.py
	rm -rf build/
