from __future__ import print_function

#=======================================================================
#           simple test for f90wrap 
#=======================================================================


import numpy as np

#=======================================================================
#the first import is a subroutine, the second is a module)
#=======================================================================

#from mockdt import *
import mockdtpkg as md
      
#=======================================================================
# call a "top-level" subroutine. This refers to subroutines that are 
# present outside of modules, and do not operate on derived types AFAIK
#=======================================================================

#This particular routine sets some numeric constants 
md.assign_constants()

print(type(md.precision))
print(md.precision.get_zero())

#=======================================================================
#Check if the subroutine worked : it modifies the "precision" module 
#     variables (assigns constants like 1,2,3, pi etc)
#=======================================================================

assert(md.precision.get_zero()   == 0.0)
assert(md.precision.get_one()    == 1.0)
assert(md.precision.get_two()    == 2.0)
assert(md.precision.get_three()  == 3.0)
assert(md.precision.get_four()   == 4.0)

assert(md.precision.get_half()   == 0.5)

#"acid" test for trailing digits, double precision: nonterminating, nonrepeating
assert(md.precision.get_pi()     == np.pi)
print('1,2,3,4 as done by subroutine are ')
print(md.precision.get_one(),md.precision.get_two(), md.precision.get_three(),\
      md.precision.get_four())

#=======================================================================
#Create "Solveroptions" derived type, defined in mod defineallproperties
#=======================================================================

Options =  md.defineallproperties.Solveroptionsdef()

print(type(Options.airframevib))
Options.airframevib = 0
Options.fet_qddot   = 1

#=======================================================================
#           Set default values for this derived type
#=======================================================================

md.set_defaults(Options)
assert(Options.airframevib           == 1)
assert(Options.fet_qddot             == 0)
assert(Options.fusharm               == 0)
assert(Options.fet_response          == 0)
assert(Options.store_fet_responsejac == 0)

print('all tests passed, OK!')

