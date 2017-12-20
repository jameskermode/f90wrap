#  f90wrap: F90 to Python interface generator with derived type support
#
#  Copyright James Kermode 2011-2018
#
#  This file is part of f90wrap
#  For the latest version see github.com/jameskermode/f90wrap
#
#  f90wrap is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  f90wrap is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with f90wrap. If not, see <http://www.gnu.org/licenses/>.
# 
#  If you would like to license the source code under different terms,
#  please contact James Kermode, james.kermode@gmail.com
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:19:03 2015

@author: David Verelst
"""

from __future__ import print_function

import numpy as np

import ExampleArray as lib

@profile
def do_array_stuff(ndata):
    xdata = np.arange(ndata)
    fdata = np.arange(ndata)
    breakarr = np.zeros((ndata,), order='F', dtype=np.float64)
    cscoef = np.zeros((4, ndata), order='F', dtype=np.float64)

    lib.library.do_array_stuff(n=ndata, x=xdata, y=fdata,
                               br=breakarr, co=cscoef)

    cscoef_np = np.zeros((4, ndata), order='F', dtype=np.float64)
    for k in range(4):
        cscoef_np[k,:] = xdata*fdata + xdata
    breakarr_np = np.zeros((ndata), order='F', dtype=np.float64)
    breakarr_np[:] = xdata/(fdata+1.0)

    print(np.allclose(breakarr_np, breakarr))
    print(np.allclose(cscoef_np, cscoef))
    p2 = breakarr

    lib.library.only_manipulate(n=ndata, array=cscoef)
    p3 = cscoef

do_array_stuff(1e6)
