#!/usr/bin/env python
import itest

REF = 1.0
TOL = 1.0e-6

obj = itest.myclass_factory.create_myclass("impl")

output = obj.get_value()
assert(abs(output-REF)<TOL)
print(f"OK: {output} == {REF}")

del obj

REF2 = 2.0
obj2 = itest.myclass_factory.create_myclass("impl2")
output2 = obj2.get_value()
assert(abs(output2-REF2)<TOL)
print(f"OK: {output2} == {REF2}")
