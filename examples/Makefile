EXAMPLES = arrayderivedtypes \
	arrays \
	arrays_fixed \
	arrays_in_derived_types_issue50 \
	cylinder \
	derivedtypes \
	elemental \
	example2 \
	extends \
	interface \
	issue32 \
	mockderivetype \
	mod_arg_clash \
	optional_args_issue53 \
	passbyreference \
	strings \
	type_bn

test:
	for example in ${EXAMPLES}; do \
	  echo "running make test in $$example" ; \
	  make -C $$example test || exit ; \
	done

clean:
	for example in ${EXAMPLES}; do \
	  echo "running make test in $$example" ; \
	  make -C $$example clean || exit ; \
	done
