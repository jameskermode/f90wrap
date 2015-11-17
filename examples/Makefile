EXAMPLES = mockderivetype example2 arrays derivedtypes \
	cylinder elemental passbyreference strings type_bn

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
