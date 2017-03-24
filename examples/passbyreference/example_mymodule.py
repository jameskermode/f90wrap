from __future__ import print_function
import mymodule
import numpy as np

tt = mymodule.Mymodule.mytype()
tt.val = 17
b=np.array(17)

print('********* BEFORE')
print('b=',b)
print('tt=',tt)
mymodule.mymodule.mysubroutine(4,b,tt)
print('********* AFTER')
print('b=',b)
print('tt=',tt)
