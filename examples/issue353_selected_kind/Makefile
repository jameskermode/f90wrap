include ../make.inc

.PHONY: all test clean

all:
	f90wrap test_selected_kind.f90 -v

test: all
	$(PYTHON) run.py

clean:
	rm -f f90wrap*.f90 .f2py_f2cmap
	rm -rf __pycache__/ build/ _build_dir/

