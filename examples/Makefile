include make.inc

EXAMPLES = arrayderivedtypes \
	arrays \
	arrays_fixed \
	arrays_in_derived_types_issue50 \
	class_names \
	cylinder \
	derivedtypes \
	elemental \
	example2 \
	extends \
	interface \
	issue105_function_definition_with_empty_lines \
	issue32 \
	keyword_renaming_issue160 \
	mockderivetype \
	mod_arg_clash \
	name_collision \
	optional_args_issue53 \
	optional_derived_arrays \
	passbyreference \
	strings \
	subroutine_contains_issue101 \
	type_bn \
	docstring \
	type_check \
	derivedtypes_procedure \
	return_array \
	string_array_input_f2py \
	optional_string \
	long_subroutine_name \
	kind_map_default \
	intent_out_size \
	output_kind \
	remove_pointer_arg \
	fortran_oo \
	issue254_getter \
	issue258_derived_type_attributes \
	issue206_subroutine_oldstyle \
	issue227_allocatable \
	issue235_allocatable_classes \
	issue297_ignored_abstract_classes \
	auto_raise_error \
	dump_package \
	method_optional \
	relative_import \
	decoded_strings \
	keep_single_interface \
	return_bool

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
	for example in ${EXAMPLES}; do \
	  echo "" ; \
	  echo "" ; \
	  echo "" ; \
	  echo "# ---------------------------------------------------" ; \
	  echo "running make test in $$example" ; \
	  make -C $$example -f Makefile.meson PYTHON=$(PYTHON) DIRECTC=$(DIRECTC) test || echo "FAILED: $$example" ; \
	done

clean_meson:
	for example in ${EXAMPLES}; do \
	  echo "running make clean in $$example" ; \
	  make -C $$example -f Makefile.meson clean || exit ; \
	done

test_meson_directc:
	$(MAKE) test_meson DIRECTC=yes

clean_meson_directc:
	$(MAKE) clean_meson
