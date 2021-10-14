#==============================================================
# Determine and set the Fortran compiler flags we want 
#==============================================================

#==============================================================
# Make sure that the default build type is RELEASE if not specified.
#==============================================================

INCLUDE(${CMAKE_MODULE_PATH}/SetCompileFlag.cmake)

#==============================================================
# Make sure the build type is uppercase
#==============================================================

STRING(TOUPPER "${CMAKE_BUILD_TYPE}" BT)

IF(BT STREQUAL "RELEASE")
    SET(CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are DEBUG, RELEASE, or TESTING."
      FORCE)
    MESSAGE(STATUS "CMAKE BUILD TYPE RELEASE SELECTED")
ELSEIF(BT STREQUAL "DEBUG")
    SET (CMAKE_BUILD_TYPE DEBUG CACHE STRING
      "Choose the type of build, options are DEBUG, RELEASE, or TESTING."
      FORCE)
    MESSAGE(STATUS "CMAKE BUILD TYE DEBUG SELECTED")
ELSEIF(BT STREQUAL "TESTING")
    SET (CMAKE_BUILD_TYPE TESTING CACHE STRING
      "Choose the type of build, options are DEBUG, RELEASE, or TESTING."
      FORCE)
    MESSAGE(STATUS "CMAKE BUILD TYPE TESTING SELECTED")
ELSEIF(NOT BT)
    SET(CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are DEBUG, RELEASE, or TESTING."
      FORCE)
    MESSAGE(STATUS "CMAKE_BUILD_TYPE not given, defaulting to RELEASE")
ELSE()
    MESSAGE(FATAL_ERROR "CMAKE_BUILD_TYPE not valid, choices are DEBUG, RELEASE, or TESTING")
ENDIF(BT STREQUAL "RELEASE")

#==============================================================
# If the compiler flags have already been set, return now
#==============================================================

IF(CMAKE_Fortran_FLAGS_RELEASE AND CMAKE_Fortran_FLAGS_TESTING AND CMAKE_Fortran_FLAGS_DEBUG)
    RETURN ()
ENDIF(CMAKE_Fortran_FLAGS_RELEASE AND CMAKE_Fortran_FLAGS_TESTING AND CMAKE_Fortran_FLAGS_DEBUG)

#==============================================================
# Determine the appropriate flags for this compiler for each build type.
# For each option type, a list of possible flags is given that work
# for various compilers.  The first flag that works is chosen.
# If none of the flags work, nothing is added (unless the REQUIRED 
# flag is given in the call).  This way unknown compiles are supported.
#==============================================================

#==============================================================
### GENERAL FLAGS ###
#==============================================================

#==============================================================
# Don't add underscores in symbols for C-compatability
#==============================================================

#SET_COMPILE_FLAG(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}"
#                 Fortran "-fno-underscoring")

#==============================================================
# There is some bug where -march=native doesn't work on Mac
#==============================================================

IF(APPLE)
    SET(GNUNATIVE "-mtune=native")
ELSE()
    SET(GNUNATIVE "-march=native")
ENDIF()

#==============================================================
# Define the operating system
#==============================================================

SET(OS ${CMAKE_SYSTEM_NAME})
SET(FC ${CMAKE_Fortran_COMPILER})

STRING(TOUPPER "${OS}" OS)
STRING(TOUPPER "${FC}" FC)

SET(Wintel FALSE)
IF(${OS} STREQUAL "WINDOWS")
   IF(${FC} MATCHES "INTEL")
      SET(Wintel TRUE)
   ENDIF()
ENDIF()

MESSAGE("The Operating System Type is " ${OS})
MESSAGE("The Fortran Compiler is      " ${FC})

#==============================================================
# add some definitions
#==============================================================

set(CMAKE_COMPILE_FLAGS -Wall -Wno-unused-dummy-argument -Wno-unused-variable -Wno-unused-label -Wno-character-truncation -fbounds-check -ffree-line-length-0 -static-libgfortran -fdefault-real-8 -ffpe-trap=invalid,zero,overflow -O0)

#set(CMAKE_COMPILE_FLAGS -fdefault-real-8)

set(CMAKE_Fortran_FLAGS_RELEASE -fdefault-real-8)