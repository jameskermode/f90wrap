from __future__ import print_function

#=======================================================================
#           simple test for f90wrap 
#=======================================================================


import numpy as np

#=======================================================================
#the first import is a subroutine, the second is a module)
#=======================================================================

from mockdt import *

#=======================================================================
# call a "top-level" subroutine. This refers to subroutines that are 
# present outside of modules, and do not operate on derived types AFAIK
#=======================================================================

#This particular routine sets some numeric constants 
assign_constants()

#=======================================================================
#Check if the subroutine worked : it modifies the "precision" module 
#variables
#=======================================================================

assert(precision.zero   == 0.0)
assert(precision.one    == 1.0)
assert(precision.two    == 2.0)
assert(precision.three  == 3.0)
assert(precision.four   == 4.0)

assert(precision.half   == 0.5)

#"acid" test for trailing digits, double precision: nonterminating, nonrepeating
assert(precision.pi     == np.pi)
print('1,2,3,4 as done by subroutine are ')
print(precision.one,precision.two, precision.three,precision.four)

#=======================================================================
#           Declare the SolverOptions derived type
#=======================================================================

Options =  Defineallproperties.Solveroptionsdef()

print(type(Options.airframevib))
Options.airframevib = 0
Options.fet_qddot   = 1

#=======================================================================
#           Set default values for this derived type
#=======================================================================

set_defaults(Options)
assert(Options.airframevib           == 1)
assert(Options.fet_qddot             == 0)
assert(Options.fusharm               == 0)
assert(Options.fet_response          == 0)
assert(Options.store_fet_responsejac == 0)

print('all tests passed, OK!')

