cmake_minimum_required(VERSION 3.0)
project(test_f90wrap)
enable_testing()
include(CTest)

list(APPEND tests
    arrayderivedtypes
    arrays
    arrays_fixed
    arrays_in_derived_types_issue50
    class_names
    cylinder
    derivedtypes
    elemental
    example2
    extends
    interface
    issue105_function_definition_with_empty_lines
    issue32
    mockderivetype
    mod_arg_clash
    optional_args_issue53
    optional_derived_arrays
    passbyreference
    strings
    subroutine_contains_issue101
    type_bn
    kind_map_default
    docstring
    return_array
    intent_out_size
    string_array_input_f2py
    type_check
    optional_string
    long_subroutine_name
    output_kind
    remove_pointer_arg
    fortran_oo
    issue254_getter
    issue258_derived_type_attributes
    issue206_subroutine_oldstyle
    issue227_allocatable
    issue235_allocatable_classes
    auto_raise_error
	  dump_package
    method_optional
    relative_import
    decoded_strings
    keep_single_interface
    return_bool
)

foreach(test ${tests})
    message(STATUS "Adding test ${test}")
    add_test(
        NAME ${test}
        COMMAND make
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/${test}"
    )
endforeach()
