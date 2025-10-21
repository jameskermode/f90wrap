#=======================================================================
#                   define the compiler names
#=======================================================================

CC          ?= gcc
F90         ?= gfortran
FC          ?= gfortran
FPP         = gfortran -E
FPP_F90FLAGS = -x f95-cpp-input -fPIC
PYTHON      = python
CFLAGS      = -fPIC
F90FLAGS    = -fPIC
FCFLAGS     = -fPIC
F90WRAP     = f90wrap

ifeq ($(origin F90), undefined)
	ifneq ($(origin FC), undefined)
		F90 = $(FC)
	endif
endif

ifeq ($(notdir $(F90)),gfortran)
	F2PY_FCOMP    = gfortran
else ifeq ($(notdir $(F90)),nvfortran)
	F2PY_FCOMP    = nv
else ifeq ($(notdir $(F90)),ifort)
    F2PY_FCOMP    = intelem
    F90FLAGS     += -fpscomp logicals
endif

F2PYFLAGS = --build-dir build --fcompiler=$(F2PY_FCOMP)

ifeq ($(notdir $(CC)),nvc)
	CC       = nvc -noswitcherror
endif
