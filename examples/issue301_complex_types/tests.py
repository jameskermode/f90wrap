"""Test complex types work correctly after f90wrap wrapping.

See issue #301: https://github.com/jameskermode/f90wrap/issues/301
"""
import complex_mod

# Test that we can call the wrapped complex functions
z = complex_mod.complex_mod.set_complex()
re, im = complex_mod.complex_mod.get_parts(z)
assert abs(re - 1.0) < 1e-10, f"Expected re=1.0, got {re}"
assert abs(im - 2.0) < 1e-10, f"Expected im=2.0, got {im}"

a = complex(1.0, 2.0)
b = complex(3.0, 4.0)
c = complex_mod.complex_mod.add_complex(a, b)
assert abs(c.real - 4.0) < 1e-10, f"Expected c.real=4.0, got {c.real}"
assert abs(c.imag - 6.0) < 1e-10, f"Expected c.imag=6.0, got {c.imag}"

print("Done")
