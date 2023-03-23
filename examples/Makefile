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
	mockderivetype \
	mod_arg_clash \
	optional_args_issue53 \
	optional_derived_arrays \
	passbyreference \
	strings \
	subroutine_contains_issue101 \
	type_bn \
	docstring \
	type_check \
	derivedtypes_procedure

PYTHON = python

all: test

test:
	for example in ${EXAMPLES}; do \
	  echo "" ; \
	  echo "" ; \
	  echo "" ; \
	  echo "# ---------------------------------------------------" ; \
	  echo "running make test in $$example" ; \
	  make -C $$example PYTHON=$(PYTHON) test || exit ; \
	done

clean:
	for example in ${EXAMPLES}; do \
	  echo "running make test in $$example" ; \
	  make -C $$example clean || exit ; \
	done
