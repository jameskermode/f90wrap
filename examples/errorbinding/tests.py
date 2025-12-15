#!/usr/bin/env python
"""Minimal test for errorbinding example - verifies the module can be imported and used."""

import ExampleDerivedTypes

# Create a type with procedure
obj = ExampleDerivedTypes.Datatypes.typewithprocedure()

# Call init method
obj.init(a=1.0, n=5)

# Call info method (writes to stdout)
obj.info(lun=6)

print("OK: errorbinding test passed")
