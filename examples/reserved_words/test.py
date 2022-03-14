import reserved_words
import numpy

sd        = reserved_words.highest_level.outer_tmp.size_bn
type_DP   = type(sd.test_double)
type_SP   = type(sd.test_single)
type_real = type(sd.test_float)
print('type of DP   designation is ',type_DP)
print('type of SP   designation is ',type_SP)
print('type of real designation is ',type_real)
