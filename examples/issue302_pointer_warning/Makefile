# Issue #302: Pointer arrays silently skipped in direct-c mode
# https://github.com/jameskermode/f90wrap/issues/302
# FIXED: f90wrap now wraps pointer arrays in direct-c mode (PR #343)

NAME=pointer_mod
WRAPPER=_pointer_mod

SOURCES=pointer_mod.f90

# Test that pointer arrays are properly wrapped
test:
	@echo "Testing that f90wrap wraps pointer arrays..."
	@rm -f $(NAME).py $(NAME)_pkg f90wrap_*.f90 .f2py_f2cmap *.o *.mod
	@rm -rf build __pycache__ $(WRAPPER)*.so
	f90wrap -m $(NAME) $(SOURCES) -v
	f2py-f90wrap --build-dir build -c -m $(WRAPPER) f90wrap_*.f90 $(SOURCES)
	@if grep -q "f90wrap_container_t__array__data" f90wrap_$(NAME).f90; then \
	   echo "PASS: Pointer array wrapper generated successfully"; \
	 else \
	   echo "FAIL: Pointer array wrapper not generated - regression!"; \
	   exit 1; \
	 fi
	python tests.py

clean:
	rm -rf *.c $(NAME).py $(NAME)_pkg f90wrap_*.f90 *.o *.mod __pycache__ .f2py_f2cmap build $(WRAPPER)*.so

.PHONY: test clean
