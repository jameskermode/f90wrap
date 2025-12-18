include make.inc

EXAMPLES = \
	arrayderivedtypes \
	arrays \
	arrays_fixed \
	arrays_in_derived_types_issue50 \
	auto_raise_error \
	callback_print_function_issue93 \
	class_names \
	cylinder \
	decoded_strings \
	default_i8 \
	derived-type-aliases \
	derivedtypes \
	derivedtypes_procedure \
	docstring \
	dump_package \
	elemental \
	errorbinding \
	example2 \
	extends \
	f2py_string_input \
	fixed_1D_derived_type_array_argument \
	fortran_oo \
	intent_out_size \
	interface \
	issue105_function_definition_with_empty_lines \
	issue206_subroutine_oldstyle \
	issue227_allocatable \
	issue235_allocatable_classes \
	issue254_getter \
	issue258_derived_type_attributes \
	issue261_array_shapes \
	issue297_ignored_abstract_classes \
	issue299_directc_nested_functions \
	issue301_complex_types \
	issue302_pointer_warning \
	issue305_multiple_abstract_interfaces \
	issue306_allocatable_realloc \
	issue307_logical_array \
	issue32 \
	issue41_abstract_classes \
	keep_single_interface \
	keyword_renaming_issue160 \
	kind_map_default \
	long_subroutine_name \
	method_optional \
	mockderivetype \
	mod_arg_clash \
	name_collision \
	optional_args_issue53 \
	optional_derived_arrays \
	optional_string \
	output_kind \
	passbyreference \
	recursive_type \
	recursive_type_array \
	relative_import \
	remove_pointer_arg \
	return_array \
	return_bool \
	signature_vs_backend \
	string_array_input_f2py \
	strings \
	subroutine_args \
	subroutine_contains_issue101 \
	type_bn \
	type_check

DIRECTC ?= no

all: test

test:
	@echo "========================================" ; \
	echo "Starting test suite: DIRECTC=$(DIRECTC)" ; \
	echo "========================================" ; \
	START_TOTAL=$$(date +%s.%N) ; \
	for example in ${EXAMPLES}; do \
	  echo "" ; \
	  echo "# ---------------------------------------------------" ; \
	  echo "Testing: $$example" ; \
	  START=$$(date +%s.%N) ; \
	  make -C $$example PYTHON=$(PYTHON) DIRECTC=$(DIRECTC) test || exit ; \
	  END=$$(date +%s.%N) ; \
	  ELAPSED=$$(LC_NUMERIC=C printf "%.3f" $$(echo "$$END - $$START" | bc -l)) ; \
	  echo "Time: $$ELAPSED s" ; \
	done ; \
	END_TOTAL=$$(date +%s.%N) ; \
	TOTAL_ELAPSED=$$(LC_NUMERIC=C printf "%.3f" $$(echo "$$END_TOTAL - $$START_TOTAL" | bc -l)) ; \
	echo "" ; \
	echo "========================================" ; \
	echo "Total time: $$TOTAL_ELAPSED s" ; \
	echo "========================================"

clean:
	for example in ${EXAMPLES}; do \
	  echo "running make clean in $$example" ; \
	  make -C $$example clean || exit ; \
	done

test_directc:
	$(MAKE) test_meson DIRECTC=yes

clean_directc:
	$(MAKE) clean_meson

test_meson:
	@FAILED="" ; \
	for example in ${EXAMPLES}; do \
	  echo "" ; \
	  echo "# ---------------------------------------------------" ; \
	  echo "Testing: $$example" ; \
	  if make -C $$example -f Makefile.meson PYTHON=$(PYTHON) DIRECTC=$(DIRECTC) test; then \
	    echo "PASS: $$example" ; \
	  else \
	    echo "FAIL: $$example" ; \
	    FAILED="$$FAILED $$example" ; \
	  fi ; \
	done ; \
	echo "" ; \
	echo "========================================" ; \
	if [ -n "$$FAILED" ]; then \
	  echo "FAILURES:$$FAILED" ; \
	  echo "========================================" ; \
	  exit 1 ; \
	else \
	  echo "All tests passed" ; \
	  echo "========================================" ; \
	fi

clean_meson:
	for example in ${EXAMPLES}; do \
	  echo "running make clean in $$example" ; \
	  make -C $$example -f Makefile.meson clean || exit ; \
	done

test_meson_directc:
	$(MAKE) test_meson DIRECTC=yes

clean_meson_directc:
	$(MAKE) clean_meson
