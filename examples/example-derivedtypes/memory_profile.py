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
