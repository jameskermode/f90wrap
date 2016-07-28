import test_python
import numpy

a = numpy.ones(test_python.test_module.m, dtype=numpy.float32)
b = test_python.test_module.Test_Type2_Xn_Array()
c = test_python.test_module.Test_Type2_Xn_Array()
d = test_python.test_module.Test_Type2_Xm_Array()
e = test_python.test_module.Test_Type2_X5_Array()
f = numpy.ones(1)

test_python.test_module.test_routine4(a, b, c, d, e, f)

print(a)
print(list(b.items[i].y for i in range(len(b.items))))
print(list(c.items[i].y for i in range(len(c.items))))
print(list(d.items[i].y for i in range(len(d.items))))
print(list(e.items[i].y for i in range(len(e.items))))
print(f)


assert(all(a == numpy.array([42, 1, 1, 1, 1], dtype=numpy.float32)))
assert(b.items[1].y[1] == 42)
assert(c.items[2].y[2] == 42)
assert(d.items[3].y[3] == 42)
assert(e.items[4].y[4] == 42)
assert(f == 2)
