mockderivetype
--------------

This example contains a number of "tests" of f90wrap's functionality. To run
the tests, simply type "make" ("make clean" to restart). 

The tests that it contains include:
 * wrap a module containing a module-level variable which is a defined type
 * this defined type is defined in a separate, used, module.
 * the defined type itself references another defined type defined in another
   separate, used module.
 * the defined type also contain a logical, integer, real and vector.
 * tests that subroutines not explicitly provided are not wrapped.
 * tests that top-level subroutines are wrapped properly if provided.
 * tests that module-level variables in used modules are accessible
 * running "python test.py" after making the example tests the pythonic wrapper.
 
