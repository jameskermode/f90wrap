include ../make.meson.inc

NAME := base
PYTHON  ?= python
UNAME   = $(shell uname)
LIBSRC_WRAP_FPP_FILES := base.f90

test: build
	$(PYTHON) run.py

.PHONY: test
