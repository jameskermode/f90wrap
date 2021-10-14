#==============================================================
# define function to perform preprocessing
#==============================================================

function(preprocess filename flags subdir ext)

#==============================================================
# find filename without extensions
#==============================================================

    get_filename_component(name ${filename} NAME_WE)

#==============================================================
# Define preprocessed file name with full path (in BIN dir)
#==============================================================

    set(output ${BIN}/${subdir}/${name}${ext})

#    file(RELATIVE_PATH ${name} ${BIN})
#==============================================================
# Print some messages to the screen
#==============================================================

#    MESSAGE("FILE NAME IS " ${name} ".F90")
#    MESSAGE("DIRECTORY IS " ${path})
#    MESSAGE("OUTPUT FILE IS " ${output})
#    MESSAGE("COMPILER FLAGS ARE " ${flags})
#==============================================================
# Add a custom command for preprocessing
#==============================================================

    add_custom_command(

#==============================================================
# Define the output
#==============================================================

        OUTPUT ${output}

#==============================================================
# Command is use the fortran compiler with the flags given on 
# the filename specified, pass everything verbatim
#==============================================================

        COMMAND ${FC} ${flags} ${filename} -o ${output}
        COMMENT "PREPROCESSING FORTRAN FILE" ${filename}
        WORKING_DIRECTORY ${BIN}/${subdir}
        VERBATIM)

#==============================================================
# tag this source file (.fpp) as being a generated output of 
# another process
#==============================================================

    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endfunction()


#==============================================================
# Function to change file extension
#==============================================================

function (change_extension file extension subdir)

#==============================================================
# Get name of file (without extension) and path
#==============================================================

  GET_FILENAME_COMPONENT(name ${file} NAME_WE)

#==============================================================
# Define filename with new extension
#==============================================================

  set(file ${BIN}/${subdir}/${name}${extension})

#==============================================================
# The file name is both input and output; for CMake, we need to 
# set the scope to PARENT so that it's changed in the called 
# location!
#==============================================================

  set(file ${file} PARENT_SCOPE)

#==============================================================
# Print messages to screen
#==============================================================

#  MESSAGE("PATH IS " ${path})
#  MESSAGE("FILE NAME IS " ${file})

#==============================================================
# End of operations
#==============================================================

endfunction()

#==============================================================
# Function to identify f90wrap-specific Fortran90 flags
#==============================================================

function (id_flags compiler flags PAR_FLAG OMP_LINK CN EXT)

  GET_FILENAME_COMPONENT(name ${compiler} NAME)
  STRING(TOLOWER ${name} compiler)
  MESSAGE("COMPILER SHORT NAME IS " ${compiler})

#==============================================================
# Generate position-independent code: same for all: -fPIC
#==============================================================

#==============================================================
# gfortran
#==============================================================

  IF(${compiler} STREQUAL "gfortran" OR ${compiler} STREQUAL "gfortran.exe")
    SET(flags -E)
    set(PAR_FLAG -fopenmp)                         # parallelization flags
    set(OMP_LINK -lgomp)                           # openmp flags
    set(CN gfortran)
    set(EXT .fpp)
#==============================================================
# intel fortran
#==============================================================

  ELSEIF(${compiler} STREQUAL "ifort")
    SET(flags -P -fPIC -fpp -Dnogpu)
    set(PAR_FLAG -openmp)                          # parallelization flags
    set(OMP_LINK -liomp5)                          # openmp flags
    set(CN intelem)
    set(EXT .fpp)

#==============================================================
# PGI (GPU) fortran
#==============================================================

  ELSEIF(${compiler} STREQUAL "pgfortran")
    SET(flags -F -fPIC)
    set(PAR_FLAG -mp)                              # parallelization flags
#  set(OMP_LINK )                                # openmp flags
    set(CN pg)
    set(EXT .f90)

#==============================================================
# Unsupported option
#==============================================================

  ELSE()
    MESSAGE(FATAL ERROR "I DONT KNOW WHAT COMPILER IS SET")
  ENDIF()

  MESSAGE("COMPILER IS " ${compiler})
#==============================================================
# Set variable scopes to parent
#==============================================================

  set(flags ${flags} PARENT_SCOPE)
  set(PAR_FLAG  ${PAR_FLAG}  PARENT_SCOPE)
  set(OMP       ${OMP_LINK}  PARENT_SCOPE)
  set(CN ${CN} PARENT_SCOPE)
  set(EXT ${EXT} PARENT_SCOPE)

endfunction()
