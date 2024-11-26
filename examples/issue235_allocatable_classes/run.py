#!/usr/bin/env python
import itest

REF = 3.1415
TOL = 1.0e-14

obj = itest.myclass_factory.myclass_create(REF)

output = obj.get_val()
assert(abs(output-REF)<TOL)
print(f"OK: {output} == {REF}")
