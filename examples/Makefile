EXAMPLES = mockderivetype example2 example-arrays example-derivedtypes \
	cylinder elemental passbyreference example-strings

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
