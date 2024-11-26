#!/usr/bin/env python
import itest

REF = 1.0
TOL = 1.0e-14

obj = itest.myclass_factory.create_myclass("impl")

output = obj.get_value()
assert(abs(output-REF)<TOL)
print(f"OK: {output} == {REF}")

# obj2 = itest.myclass_factory.create_myclass("impl")
# output2 = obj2.get_value()
# assert(abs(output-REF)<TOL)
# print(f"OK: {output2} == {REF}")
