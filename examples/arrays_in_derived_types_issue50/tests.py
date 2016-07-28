from issue50 import module_test as tp
from numpy import zeros, ones, float32, abs, max

a = tp.Real_Array()
print("This is the freshly allocated array : " + str(a.item))
a.item = ones(6, dtype=float32) 
print("This is sent to fortran : " + str(a.item))
tp.testf(a)
print("This is received by python : " + str(a.item))

assert max(abs(a.item - [1.0, 1.0, 1.0, 4.0, 1.0, 1.0])) < 1e-6
