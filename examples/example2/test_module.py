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
from __future__ import print_function

#=======================================================================
#           simple test for f90wrap
#=======================================================================


import numpy as np
import re
import unittest

#=======================================================================
#the first import is a subroutine, the second is a module)
#=======================================================================

from mockdt import *
import os

#=======================================================================
# call a "top-level" subroutine. This refers to subroutines that are
# present outside of modules, and do not operate on derived types AFAIK
#=======================================================================

class TestTypeCheck(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTypeCheck, self).__init__(*args, **kwargs)

    def test_case_1(self):
        #This particular routine sets some numeric constants
        assign_constants()

        #=======================================================================
        #Check if the subroutine worked : it modifies the "precision" module
        #variables
        #=======================================================================

        self.assertEqual(precision.zero, 0.0)
        self.assertEqual(precision.one, 1.0)
        self.assertEqual(precision.two, 2.0)
        self.assertEqual(precision.three, 3.0)
        self.assertEqual(precision.four, 4.0)

        self.assertEqual(precision.half, 0.5)

        #"acid" test for trailing digits, double precision: nonterminating, nonrepeating
        self.assertEqual(precision.pi, np.pi)
        print('1,2,3,4 as done by subroutine are ')
        print(precision.one,precision.two, precision.three,precision.four)

        #=======================================================================
        #           Declare the SolverOptions derived type
        #=======================================================================

    @unittest.skipIf(re.search("nvfortran", os.environ.get('F90', '')), "Fails with nvfortran")
    def test_case_2(self):
        Options =  Defineallproperties.SolverOptionsDef()

        print(type(Options.airframevib))
        Options.airframevib = 0
        Options.fet_qddot   = 1

        #=======================================================================
        #           Set default values for this derived type
        #=======================================================================

        set_defaults(Options)
        self.assertEqual(Options.airframevib, 1)
        self.assertEqual(Options.fet_qddot, 0)
        self.assertEqual(Options.fusharm, 0)
        self.assertEqual(Options.fet_response, 0)
        self.assertEqual(Options.store_fet_responsejac, 0)

if __name__ == '__main__':

    unittest.main()
