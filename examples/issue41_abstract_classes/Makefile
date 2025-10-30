include ../make.inc

all: test test_abstract_classes.x

test: wrapper
	$(PYTHON) run.py

test_abstract_classes.x: main.f90 myclass_base.o myclass_impl.o myclass_impl2.o myclass_factory.o
	$(F90) $(FCFLAGS) -o $@ $^

wrapper: myclass_base.o myclass_impl.o myclass_impl2.o myclass_factory.o
	f90wrap -m itest -P myclass_base.f90 myclass_impl.f90 myclass_impl2.f90 myclass_factory.f90 -v
	f2py-f90wrap $(F2PYFLAGS) -c -m _itest --opt="-O0 -g" \
		f90wrap_myclass_base.f90 f90wrap_myclass_impl.f90 f90wrap_myclass_impl2.f90 f90wrap_myclass_factory.f90 \
		myclass_base.o myclass_impl.o myclass_impl2.o myclass_factory.o

%.o : %.f90
	$(F90) $(FCFLAGS) -c -g -O0 $< -o $@

clean:
	rm -f *.o f90wrap*.f90 *.so *.mod *.x
	rm -rf src.*/
	rm -rf itest/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
