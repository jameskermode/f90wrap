#!/usr/bin/env python3

version = {}
with open('f90wrap/__init__.py') as fp:
    exec(fp.read(), version)
__version__ = version['__version__']

print(__version__)
