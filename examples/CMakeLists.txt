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
    default_i8
    derived-type-aliases
    derivedtypes
    derivedtypes_procedure
    docstring
    dump_package
    elemental
    errorbinding
    example2
    extends
    fixed_1D_derived_type_array_argument
    fortran_oo
    intent_out_size
    interface
    issue105_function_definition_with_empty_lines
    issue206_subroutine_oldstyle
    issue227_allocatable
    issue235_allocatable_classes
    issue254_getter
    issue258_derived_type_attributes
    issue261_array_shapes
    issue297_ignored_abstract_classes
    issue301_complex_types
    issue302_pointer_warning
    issue305_multiple_abstract_interfaces
    issue306_allocatable_realloc
    issue307_logical_array
    issue32
    issue41_abstract_classes
    keep_single_interface
    keyword_renaming_issue160
    kind_map_default
    long_subroutine_name
    method_optional
    mockderivetype
    mod_arg_clash
    name_collision
    optional_args_issue53
    optional_derived_arrays
    optional_string
    output_kind
    passbyreference
    recursive_type_array
    relative_import
    remove_pointer_arg
    return_array
    return_bool
    signature_vs_backend
    string_array_input_f2py
    strings
    subroutine_args
    subroutine_contains_issue101
    callback_print_function_issue93
    f2py_string_input
    issue299_directc_nested_functions
    recursive_type
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
