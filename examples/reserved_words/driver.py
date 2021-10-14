import test
import numpy

sd        = test.highest_level.outer_tmp.size_bn
type_DP   = type(sd.test_double)
type_SP   = type(sd.test_single)
type_real = type(sd.test_float)
print('type of DP   designation is ',type_DP)
print('type of SP   designation is ',type_SP)
print('type of real designation is ',type_real)

N=5
rcond=0.0
determ=0.0
atemp = numpy.asfortranarray(numpy.zeros((N,N)))
for i in range(N):
    atemp[i,i] = i+1

test.invert(atemp,N,N,rcond,determ)

print(atemp)