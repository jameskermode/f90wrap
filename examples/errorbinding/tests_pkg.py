#!/usr/bin/env python
"""Minimal test for errorbinding example (package mode) - verifies the module can be imported and used."""

import ExampleDerivedTypes_pkg

# Create a type with procedure
obj = ExampleDerivedTypes_pkg.datatypes.typewithprocedure()

# Call init method
obj.init(a=1.0, n=5)

# Call info method (writes to stdout)
obj.info(lun=6)

print("OK: errorbinding package test passed")
