set(prj test)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN}/${prj})

# all sources to create executable
SET(all_src "")

file(GLOB pyint_files_src *.F90)

# add files to list of sources
LIST(APPEND all_src ${pyint_files_src})

# include preprocessor function from include file
INCLUDE(${CMAKE_MODULE_PATH}/f90wrap.cmake) 

# Choose compiler and set flags
GET_FILENAME_COMPONENT(CN ${CMAKE_Fortran_COMPILER} NAME)
MESSAGE("COMPILER IS " ${CN})
MESSAGE("ALLSRC ARE" ${all_src})
INCLUDE(${CMAKE_MODULE_PATH}/set_compiler_flags.cmake)

# Build NDARC
# 1. list of source files that need to be wrapped (APIs)
SET(lib_src "")
LIST(APPEND lib_src ${all_src})

# 2. Create user-defined libraries: list libraries & locations
MESSAGE("PROPERTIES ARE" ${PROPERTIES})
MESSAGE("COMPILE_FLAGS" ${COMPILE_FLAGS})
# SET_TARGET_PROPERTIES(ndarc_lib PROPERTIES COMPILE_FLAGS ${OpenMP_Fortran_FLAGS})
set(UDP ${LIB})

# set variable called UDL (user defined library)
set(UDL fortran_routines_${prj})
add_library(${UDL} ${lib_src})

# no flags for lapack routines
target_compile_options(${UDL} PRIVATE ${CMAKE_COMPILE_FLAGS})

# set fortran compiler for cmake
set(FC ${CMAKE_Fortran_COMPILER})

GET_FILENAME_COMPONENT(CN ${CMAKE_Fortran_COMPILER} NAME)

set(UDLs ${UDL})
set(UDPs ${UDP})
#-----------------------------
# 3. Run f90wrap: cmake script
#-----------------------------
# arguments: 
#   1. FC       = fortran compiler
#   2. FFLAGS   = fortran compilation flags
#   3. src_list = which files to wrap
#   4. prj      = project name 
#   5. mod      = location of modules
#   6. udl      = name of library with dependencies
#   7. udp      = locations of libraries
#   8. pyd      = where to place compiled binaries

# 3a. define list of sources for which f90wrap must generate APIs
set(src_list ${pyint_files_src})

MESSAGE("CMAKE FLAGS PASSED TO F90WRAP ARE " ${CMAKE_COMPILE_FLAGS})
# 3b. call wrapper
run_f90wrap("${FC}" "${CMAKE_COMPILE_FLAGS}" "${src_list}" "${prj}" "${MOD}" "${UDLs}" "${UDPs}" "${PYD}")