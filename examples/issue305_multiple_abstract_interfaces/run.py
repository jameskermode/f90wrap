#!/usr/bin/env python
import itest


obj = itest.main.myclass_impl_t()

REF = 1.0
TOL = 1.0e-6
output = obj.get_value()
assert (output - REF) < TOL
print(f"OK: {output} == {REF}")

REF = 42
output = obj.get_value2()
assert output == REF
print(f"OK: {output} == {REF}")
