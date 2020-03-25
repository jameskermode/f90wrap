#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

import numpy as np
import itest

a = np.arange(10, dtype=np.float32)
itest.itestit.testit1(a)
print(a)

itest.itestit.testit2(a)
print(a)
