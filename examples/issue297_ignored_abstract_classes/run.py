#!/usr/bin/env python
import itest

REF = 1.0
TOL = 1.0e-6

output = itest.main.use_myclass()

assert(abs(output-REF)<TOL)
print(f"OK: {output} == {REF}")
