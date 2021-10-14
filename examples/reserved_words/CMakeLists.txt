project(NDARC)
cmake_minimum_required(VERSION 3.0)
enable_language(Fortran)

set(NDARC_MAJOR_VERSION 0)
set(NDARC_MINOR_VERSION 2)             
set(NDARC_VERSION ${NDARC_MAJOR_VERSION}.${NDARC_MINOR_VERSION})

# Require fortran90 compiler
IF(NOT CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    MESSAGE(FATAL_ERROR "Fortran compiler does not support F90")
ENDIF(NOT CMAKE_Fortran_COMPILER_SUPPORTS_F90)

# Add local modules to the module path
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

INCLUDE(${CMAKE_MODULE_PATH}/SetFortranFlags.cmake) 
INCLUDE(${CMAKE_MODULE_PATH}/set_compiler_flags.cmake) 

# show compile definitions
get_directory_property( DirDefs COMPILE_DEFINITIONS )
message( "COMPILE_DEFINITIONS = ${DirDefs}" )

MESSAGE("NDARC MAIN: CMAKE FLAGS ARE" ${CMAKE_Fortran_FLAGS_RELEASE})

# No openMP or MPI allowed 
set(USE_OPENMP FALSE)
MESSAGE("USING OPENMP?" ${USE_OPENMP})
set(USE_MPI FALSE)

set(PAR_TAG (${USE_MPI} OR ${USE_OPENMP}))
IF(${PAR_TAG})
	INCLUDE(${CMAKE_MODULE_PATH}/SetParallelizationLibrary.cmake)
ENDIF(${PAR_TAG})
MESSAGE("openmp flags are" ${OPENMP_Fortran_FLAGS})

# unset some legacy flags; dont link executable to shared libraries
UNSET(CMAKE_SHARED_LIBRARY_LINK_Fortran_FLAGS)

# offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

# make relative paths absolute
foreach(p LIB BIN INCLUDE CMAKE)
   set(var INSTALL_${p}_DIR)
   if(NOT IS_ABSOLUTE "${${var}}")
      set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
   endif()
endforeach()

# Define some directories
SET(SRC ${CMAKE_SOURCE_DIR}/src)
set(BLD "cmake/build")
SET(LIB "cmake/build/bin")
SET(BIN "cmake/build/bin")
set(MOD "cmake/build/modules")
set(EXD ${CMAKE_SOURCE_DIR})
set(PYD ${EXD})


GET_FILENAME_COMPONENT(LIB ${LIB} ABSOLUTE)
GET_FILENAME_COMPONENT(BIN ${BIN} ABSOLUTE)
GET_FILENAME_COMPONENT(MOD ${MOD} ABSOLUTE)
GET_FILENAME_COMPONENT(PYD ${PYD} ABSOLUTE)
GET_FILENAME_COMPONENT(EXD ${EXD} ABSOLUTE)
GET_FILENAME_COMPONENT(BLD ${BLD} ABSOLUTE)

#==============================================================
# Set directories
#==============================================================

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIB})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIB})
set(CMAKE_BINARY_DIR ${BIN})

MESSAGE("DETERMINED DIRECTORIES")

#==============================================================
# Set cmake policy to ignore nonsense policy: silly fellows
# Tested on CMake 3.2 on Ubuntu 14.04, 3.6 on Mac 10.10.5, 
#==============================================================

MESSAGE("SETTING CMAKE POLICY 0039 BEHAVIOR TO OLD")
cmake_policy(SET CMP0039 OLD)

# The source for the FOO binary and have it placed in the bin folder
# add subdirectories
MESSAGE("ADDING SUBDIR" ${SRC})
add_subdirectory(${SRC})
# add_subdirectory(common/src/fea/)
# add_subdirectory(common/src/bemt/)

#####################################################################
# END OF FILE
#####################################################################
