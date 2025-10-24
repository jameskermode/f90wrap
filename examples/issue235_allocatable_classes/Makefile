include ../make.inc

all: test

test: wrapper
	$(PYTHON) run.py

wrapper: f90wrapper mytype.o myclass.o myclass_factory.o
	$(PYTHON) -m f90wrap --f2py-f90wrap ${F2PYFLAGS} -c -m _itest --opt="-O0 -g" \
		f90wrap_mytype.f90 f90wrap_myclass.f90 f90wrap_myclass_factory.f90 \
		mytype.o myclass.o myclass_factory.o --lower

f90wrapper: mytype.f90 myclass.f90 myclass_factory.f90
	$(PYTHON) -m f90wrap -m itest mytype.f90 myclass.f90 myclass_factory.f90 -v

%.o : %.f90
	$(F90) $(FCFLAGS) -c -g -O0 $< -o $@

clean:
	rm -f *.o f90wrap*.f90 *.so *.mod
	rm -rf src.*/
	rm -rf itest.py
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
