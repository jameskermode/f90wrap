#==============================================================
# Function to run f90wrap on a list of source files
#==============================================================

function (run_f90wrap compiler F90FLAGS_in src_list proj moddir libs locs outpath)

#==============================================================
# Find the operating system
#==============================================================

  STRING(TOUPPER "${OS}" OS)

  IF(${OS} STREQUAL "WINDOWS")
    set(windows TRUE)
  ELSE()
    set(windows FALSE)
  ENDIF()

  MESSAGE("WINDOWS OR NOT? " ${windows})

#==============================================================
# First step: make directory to place f90wrap-generated files
#==============================================================

  SET(subdir f90wrap)

# inject a space character between flags
  SET(F90FLAGS "--f90flags=")
  FOREACH (flag ${F90FLAGS_in})
    list(APPEND F90FLAGS ${flag})
    list(APPEND F90FLAGS " ")
  ENDFOREACH(flag)
  string(CONCAT F90FLAGS ${F90FLAGS} "")
  MESSAGE("F90FLAGS ARE" ${F90FLAGS})
  SET(path ${BIN}/${proj})
  IF(EXISTS ${path})
  ELSE()
    FILE(MAKE_DIRECTORY ${path})
  ENDIF(EXISTS ${path})

  set(wrapdir ${path}/${subdir})
  IF(EXISTS ${wrapdir})
  ELSE()
    FILE(MAKE_DIRECTORY ${wrapdir})
  ENDIF(EXISTS ${wrapdir})

#==============================================================
# Include nested function definitions
#==============================================================

  INCLUDE(${CMAKE_MODULE_PATH}/Preprocess_definition.cmake) 

#==============================================================
# Identify compiler and set flags
#==============================================================
  
  MESSAGE(${compiler} " is the compiler")
  id_flags("${compiler}" "${flags}" "${PAR_FLAG}" "${OMP}" "${CN}" "${EXT}")

  IF(NOT USE_OPENMP AND ${OS} STREQUAL "WINDOWS")
    SET(PAR_FLAG " ")
    SET(OMP " ")
  ENDIF()
#==============================================================
# Build list of preprocessed file names
#==============================================================

  set(file_list "")
  set(subfolder ${proj}/${subdir})
  set(flags2 ${flags})
  
#==============================================================
# loop over files that need to be integrated with python
#==============================================================

  MESSAGE("FLAGS" ${flags2})
  FOREACH(filename ${src_list})

#==============================================================
# Run the preprocessor using the compiler
#==============================================================

    preprocess(${filename} "${flags2}" "${subfolder}" "${EXT}")  

#==============================================================
# copy the filename into a temp variable
# (dunno if you can change cmake iterators)
#==============================================================

    set(file ${filename})

#==============================================================
# Change the extension and add it to a new list
#==============================================================

    change_extension(${file} ${EXT} "${subfolder}")
    LIST(APPEND file_list ${file})      # preprocessed files

  ENDFOREACH(filename)

#==============================================================
# Add custom target (dummy tag PREPROC) that tells CMake that 
# ${file_list} is the list of dependencies needed to make it 
# Without this custom target, the preprocessor will not run
#==============================================================

  add_custom_target(
    PREPROC_${prj} 
    DEPENDS ${file_list}
    COMMENT "Preprocessing file"
    VERBATIM)

#==============================================================
# Now create CMake commands to run f90wrap
#==============================================================

#==============================================================
# Step 2: run f90wrap on source files to create API defs in .py
#==============================================================

  set(output ${path}/${proj}.py)
  set(outpt2 ${path}/${proj})
#MESSAGE("OUTPUT OF STAGE 1 F90wrap is " ${output})
  set(kmap_file ${CMAKE_MODULE_PATH}/kind_map)

#==============================================================
# Add a custom command for running f90wrap
#==============================================================

  IF (${windows})
    add_custom_command(
      OUTPUT ${output}
      COMMAND python "F:\\Programs\\Miniconda\\Scripts\\f90wrap" -m ${proj} ${file_list} -k ${kmap_file} -v
#      COMMAND python "C:\\Users\\Ananth\\Anaconda3\\Scripts\\f90wrap" -m ${proj} ${file_list} -k ${kmap_file} -v
#      COMMAND python "C:\\Python27\\Scripts\\f90wrap" -m ${proj} ${file_list} -k ${kmap_file} -v
      COMMENT "CREATING PYTHON MODULE" ${output}
      WORKING_DIRECTORY ${path}
      )
  ELSE()
    add_custom_command(
      OUTPUT ${output}
      COMMAND f90wrap -m ${proj} ${file_list} -k ${kmap_file} -v
      COMMENT "CREATING PYTHON MODULE" ${output}
      WORKING_DIRECTORY ${path}
      )
  ENDIF()

#==============================================================
# Add list of libraries and locations for f90wrap visibility
#==============================================================

#  MESSAGE("file with list of f90wrap generated sources: " ${txtfile})
#  set(txtfile ${path}/f90wrap_list.txt)

#==============================================================
# Windows version: use PowerShell command to write to file
#==============================================================

  # IF(${windows})
  #   add_custom_command(
  #     OUTPUT ${txtfile} 
  #     COMMENT "SAVED LIST OF F90WRAP GENERATED FILES IN  $txtfile$"
  #     COMMAND "Get-ChildItem f90wrap_*.f90| ForEach-Object { $_.FullName } > f90wrap_list.txt"
  #     WORKING_DIRECTORY ${path}
  #   )

#==============================================================
# Linux/Mac version: use realpath command
#==============================================================

  # ELSE()
  #   add_custom_command(
  #     OUTPUT ${txtfile} 
  #     COMMENT "SAVED LIST OF F90WRAP GENERATED FILES IN " ${txtfile}
  #     COMMAND realpath f90wrap_*.f90 > f90wrap_list.txt
  #     WORKING_DIRECTORY ${path}
  #   )
  # ENDIF()  

#==============================================================
# f90wrap-created python API is target
#==============================================================
  
  LIST(GET libs 0 first_lib)      # dependency is first library 
  add_custom_target(
    API_${prj} ALL
    DEPENDS PREPROC_${prj} ${output} ${file_list} ${first_lib}
    COMMENT "creating python API"
    VERBATIM)

#==============================================================
# Step 3: run f2py-f90wrap to create shared object
#==============================================================

  IF(${windows})
    set(SO _${proj}.pyd)
  ELSE()
    set(SO _${proj}.so)                                   # shared object name
  ENDIF()
  MESSAGE("DIRECTORY IS " ${wrapdir})                # print message

#==============================================================
# Read the list of compiled f90wrap_*.f90 files 
#==============================================================

  # add_custom_command(
  #   OUTPUT f90src_${prj}
  #   DEPENDS ${txtfile}
  #   execute_process (
  #     COMMAND bash -c IFS=$'\\n' read -d '' -a lines < ${txtfile}
  #     COMMAND bash -c echo $lines[@]
  #   )
  #   SET(f90src_${prj} ${lines})
  #   MESSAGE("OUTPUT IS " ${lines}) 
  #   WORKING_DIRECTORY ${path}
  #   )

#==============================================================
# TARGET HANDLE IS f90read_${prj}, which depends on reading the
# list of files with f90wrap_*.f90 prefix 
#==============================================================

#  add_custom_target(
#    f90read_${prj} ALL
#    DEPENDS f90src_${prj}
#    )

#==============================================================
# Find all files that match the pattern f90wrap_*.f90 
#==============================================================

  IF(${windows})
  ELSE()
    set(f90src "f90wrap_*.f90")
  ENDIF()
#==============================================================
# First check whether #libraries <= #paths
#==============================================================

  LIST(LENGTH libs len)
  LIST(LENGTH locs ln2)
  message("LENGTH OF LIST IS " ${len} "," ${ln2})

  if(${len} GREATER ${ln2} AND ${len} GREATER 0)
    MESSAGE("CRITICAL ERROR")
    MESSAGE(${len} " LIBRARIES ARE " ${libs})
    MESSAGE(${ln2} " LOCATIONS ARE " ${locs})
    MESSAGE(FATAL ERROR "CANNOT PROCEED: number of libraries exceeds number of paths ")
  endif()

#==============================================================
# IF error trap is not triggered, create list of library links
#==============================================================

  set(f90wrap_links "")
  MATH(EXPR ln2 "${ln2}-1")
  MESSAGE("list length is " ${ln2})
  foreach(val RANGE ${ln2})
    list(GET libs ${val} lib_name)
    list(GET locs ${val} lib_path)
    message(STATUS "ADDING LIBRARY ${lib_name} IN LOCATION \n ${lib_path}")
    IF(${windows})
      LIST(APPEND f90wrap_links "-L${lib_path} -l${lib_name}")
    ELSE()
      LIST(APPEND f90wrap_links -L${lib_path} -l${lib_name})
    ENDIF()
  endforeach()
  MESSAGE("links" ${f90wrap_links})
#==============================================================
# Fortran compilation options
#==============================================================

  IF(${windows})
    FILE(WRITE "${path}\\final_part1.txt" "python F:\\Programs\\Miniconda\\Scripts\\f2py-f90wrap -c -m _${proj} ${f90wrap_links}")
#    FILE(WRITE "${path}\\final_part1.txt" "python C:\\Users\\Ananth\\Ananconda3\\Scripts\\f2py-f90wrap -c -m _${proj} ${f90wrap_links}")
#    FILE(WRITE "${path}\\final_part1.txt" "python C:\\Python27\\Scripts\\f2py-f90wrap -c -m _${proj} ${f90wrap_links}")
    set(f90wrap_options "--fcompiler=${CN} ${F90FLAGS} -I${moddir} --build-dir .")
    FILE(WRITE "${path}\\final_part3.txt" "${f90wrap_options}")
  ELSE()
    set(f90wrap_options --fcompiler=${CN} ${F90FLAGS} -I${moddir} --build-dir .)
    MESSAGE('f90wrap_options are ' ${f90wrap_options})
    LIST(APPEND f90wrap_links ${f90src})
  ENDIF()

# create shared object with f2py-f90wrap command
  add_custom_command(
    OUTPUT ${SO}        # Define the output
    COMMENT ("CREATING SHARED OBJECT" ${SO})
    COMMAND   f2py-f90wrap -c -m _${proj} ${f90wrap_links} ${f90wrap_options} #-v
    WORKING_DIRECTORY ${path}
  )

  add_custom_target(
    SHAREDOBJECT_${prj} ALL
    DEPENDS ${SO} API_${prj} 
    COMMENT "creating shared object"
    VERBATIM)

#==============================================================
# Link libraries to target
#==============================================================

#  TARGET_LINK_LIBRARIES(SHAREDOBJECT_${prj} ${libs})

#==============================================================
  
  set(PY ${proj}.py)
  add_custom_command(TARGET SHAREDOBJECT_${prj}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy "_*.so" ${outpath}
                   WORKING_DIRECTORY ${path})

  add_custom_command(TARGET API_${prj}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${PY} ${outpath}
                   WORKING_DIRECTORY ${path})

  MESSAGE("COPIED FILES TO " ${outpath})
endfunction()