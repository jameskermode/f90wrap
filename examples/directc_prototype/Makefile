# Test case for direct-c mode with interface prototypes
# This tests the fix for GitHub issue #358

all: test

test:
	f90wrap -m test_prototype --direct-c test_module.f90
	@echo "Success: direct-c mode handles interface prototypes correctly"

clean:
	rm -f f90wrap_*.f90 *.c *.py .f2py_f2cmap

.PHONY: all test clean
