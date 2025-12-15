cmake_minimum_required(VERSION 3.0)
project(test_f90wrap)
enable_testing()
include(CTest)

list(APPEND tests
    arrayderivedtypes
    arrays
    arrays_fixed
    arrays_in_derived_types_issue50
    auto_raise_error
    class_names
    cylinder
    decoded_strings
    derivedtypes
    docstring
    dump_package
    elemental
    example2
    extends
    fortran_oo
    intent_out_size
    interface
    issue105_function_definition_with_empty_lines
    issue206_subroutine_oldstyle
    issue227_allocatable
    issue235_allocatable_classes
    issue254_getter
    issue258_derived_type_attributes
    issue297_ignored_abstract_classes
    issue305_multiple_abstract_interfaces
    issue307_logical_array
    issue32
    keep_single_interface
    kind_map_default
    long_subroutine_name
    method_optional
    mockderivetype
    mod_arg_clash
    optional_args_issue53
    optional_derived_arrays
    optional_string
    output_kind
    passbyreference
    relative_import
    remove_pointer_arg
    return_array
    return_bool
    string_array_input_f2py
    strings
    subroutine_contains_issue101
    type_bn
    type_check
)

foreach(test ${tests})
    message(STATUS "Adding test ${test}")
    add_test(
        NAME ${test}
        COMMAND make
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/${test}"
    )
endforeach()
