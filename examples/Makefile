EXAMPLES = mockderivetype example2 example-arrays example-derivedtypes cylinder elemental passbyreference

test:
	for example in ${EXAMPLES}; do \
	  echo "running make test in $$example" ; \
	  make -C $$example test ; \
	done
