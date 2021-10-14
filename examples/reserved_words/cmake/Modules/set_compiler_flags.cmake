#==============================================================
# Reset options for different fortran compilers
#==============================================================

#==============================================================
# Uncomment if it is required that Fortran 90 is supported
#==============================================================

IF(NOT CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    MESSAGE(FATAL_ERROR "Fortran compiler does not support F90")
ENDIF(NOT CMAKE_Fortran_COMPILER_SUPPORTS_F90)

#==============================================================
# Uncomment the below if you want the user to choose a
# parallelization library
#==============================================================

#OPTION(USE_MPI "Use the MPI library for parallelization" OFF)
#OPTION(USE_OPENMP "Use OpenMP for parallelization" OFF)

#FindOpenMP()

#==============================================================
# Have the .mod files placed in a separate folder
#==============================================================

SET(CMAKE_Fortran_MODULE_DIRECTORY ${MOD})

#==============================================================
# This INCLUDE statement executes code that sets the compile
# flags for DEBUG, RELEASE, and TESTING.  You should  review
# this file and make sure the flags are to your liking.
#==============================================================

INCLUDE(${CMAKE_MODULE_PATH}/SetFortranFlags.cmake) 

#==============================================================
# Locate and set parallelization libraries.  There are some
# CMake peculiarities taken care of here, such as the fact that
# the FindOpenMP routine doesn't know about Fortran.
#==============================================================

INCLUDE(${CMAKE_MODULE_PATH}/SetParallelizationLibrary.cmake)

#==============================================================
# There is an error in CMAKE with this flag for pgf90.  Unset it
#==============================================================

#GET_FILENAME_COMPONENT(FCNAME ${CMAKE_Fortran_COMPILER} NAME)
#IF(FCNAME STREQUAL "pgfortran")
    UNSET(CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS)
#ENDIF(FCNAME STREQUAL "pgfortran")

#MESSAGE("Before auto-removing O3 flag" ${CMAKE_Fortran_FLAGS_RELEASE})
