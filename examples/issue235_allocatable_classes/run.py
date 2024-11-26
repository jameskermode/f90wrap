#!/usr/bin/env python
import itest
import itest.myclass

REF = 3.1415
TOL = 1.0e-6


create_count = itest.myclass.get_create_count()
destroy_count = itest.myclass.get_destroy_count()
assert(create_count == 0)
assert(destroy_count == 0)
print(f"OK: create_count == 0, destroy_count == 0 before creation")

obj = itest.myclass_factory.myclass_create(REF)

create_count = itest.myclass.get_create_count()
destroy_count = itest.myclass.get_destroy_count()
assert(create_count == 1)
assert(destroy_count == 0)
print(f"OK: create_count == 1, destroy_count == 0 after creation")


output = obj.get_val()
assert(abs(output-REF)<TOL)
print(f"OK: {output} == {REF}")


del obj

create_count = itest.myclass.get_create_count()
destroy_count = itest.myclass.get_destroy_count()
assert(create_count == 1)
assert(destroy_count == 1)
print(f"OK: create_count == 1, destroy_count == 1 after destruction")
