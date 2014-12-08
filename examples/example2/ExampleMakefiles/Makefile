#========================================================================
#                       Makefile for Linux
# written by Ananth Sridharan ( sridharan.ananth@gmail.com )  
#     Tested on Ubuntu 12.04, 12.10, 13.04, 13.10, 14.04. 
#                      Last updated 9/26/2014
#========================================================================
# directory for modules
#========================================================================

moddir:= ../Source/modules

#========================================================================
#      Uncomment line to choose which fortran compiler to use
#========================================================================

#F90    = gfortran
F90    = ifort
#F90    = pgfortran
AR     = ar 
AROPT  = src

#========================================================================
#                       gfortran flags
#========================================================================

ifeq ($(F90),gfortran)
	INC_FLAGS =#-Wl,-stack_size,0x40000000
	OPT_FLAGS = -llapack -lblas -J$(moddir) -fPIC
	DEB_FLAGS = #-fbounds-check -fbacktrace
	PAR_FLAGS = -fopenmp -cpp -Dopenmp -cpp -Dnogpu
	WK_FLAGS  =  
	HE_FLAGS  = -ffree-form
	HE2FLAGS  = -ffixed-line-length-132 
endif

#========================================================================
#                       IFORT flags
#========================================================================

ifeq ($(F90),ifort)
	INC_FLAGS = #-Wl,-stack_size,0x20000000  # USE THIS FOR MACS!!
	OPT_FLAGS = -llapack -lblas -module $(moddir) -fPIC
	DEB_FLAGS = #-check bounds -traceback
	PAR_FLAGS = -fpp -Dnogpu -openmp -fpp -Dopenmp 
	WK_FLAGS  = -extend_source -132
	HE_FLAGS  = -free
	HE2FLAGS  = -fixed
endif

#========================================================================
#                       PGFortran flags
#========================================================================

ifeq ($(F90),pgfortran)
	F90       = pgfortran
	INC_FLAGS = #-Wl,-z,muldefs 
	OPT_FLAGS = -module $(moddir) -llapack -lblas -fPIC 
	DEB_FLAGS = #-Mbounds -traceback #-Mchkstk Ktrap=unf   #==> for PGFortran
	PAR_FLAGS = -ta=nvidia -Minfo=accel -Mcuda=cc2x -mp=nonuma -Dopenmp \
	             -fast 
	WK_FLAGS  = -Mextend
	HE_FLAGS  = -Mfree
	HE2FLAGS  = -Mnofree
endif

#========================================================================

PRO_FLAGS = #-g

#========================================================================
#These are debugging flags.

# traceback     : indicates where the program crashed/encountererd an error,
# -check bounds : tells you whether you're trying to access elements of an 
#                 array that dont exist
# fpe0          : warns you when the result of floating point operations
#		      cannot be represented accurately in a machine
# -g 	          : debugging flag for use with a debugging program (e.g. gdb)
# -warn uninit  : Compile-time warning flag to find variables that are 
#			used before being initialized

#========================================================================
# These are parallelization flags
# -openmp       : uses openMP for multi-threaded code running on same machine
# -fpp          : is the fortran preprocessor that chooses to include or 
#		      exclude certain lines of code based on the next option
# -Dopenmp      : if openmp is used during compilation/linking, then the 
#		      statements under "#ifdef openmp" are also compiled
#                 into executable code
# -WL,...       : to use bigger stacksize on MAC architecture 
#                 (max is 8MB default)
#                 (LD_FLAGS are another way of doing this)
# -Dnogpu       : suppress CUDA-Fortran GPU wake parallelization
#                 to be used with ifort/gfortran (no support)
#
#========================================================================
#Code profiling flags
#========================================================================
# -pg           : allows for code profiling and creating a flowchart of 
#			code heirarchy using gprof and gprof2dot. The file gmon.out
#			is created after successful execution (without errors) of code
#			which can then be used with gprof 
#========================================================================
#Ok, eliminate some routines and reshuffle to keep things organized.
#let's distinguish between netlib, and UMD-written code by separating 
#code segments into directories.
#========================================================================

F90FLAGS = $(INC_FLAGS) $(OPT_FLAGS) $(PAR_FLAGS) $(DEB_FLAGS) $(PRO_FLAGS)

#========================================================================
#                     Directories definitions
#========================================================================

src:= ../Source/HeliSrc
cnd:= ../Source/MathOps
wke:= ../Source/Wake
cpl:= ../Source/CFD_CSD
bsc:= ../Source/BasicDefs
bit:= ../Source/beamdynamics
py1:= ../Source/pyint_modules
py2:= ../Source/pyint_files

#========================================================================
#         Generate modules needed for python interface 
#========================================================================

PY1_FILES:= $(wildcard $(py1)/*.F90)
PY1_OBJTS:= $(PY1_FILES:.F90=.o)

#Compilation command follows
$(py1)/%.o: $(py1)/%.F90
	$(F90) $(F90FLAGS) $(HE_FLAGS) -c -o $@ $<

#========================================================================
#                            Basic operations
#========================================================================

BSC_FILES:= $(wildcard $(bsc)/*.F90)
BSC_OBJTS:= $(BSC_FILES:.F90=.o)

#Compilation command follows
$(bsc)/%.o: $(bsc)/%.F90
	$(F90) $(F90FLAGS) $(HE_FLAGS) -c -o $@ $<
 
#========================================================================
#           Math operations: source files and objects
#========================================================================
#  HIGHLY RECOMMENDED THAT YOU DO NOT MESS WITH FILES IN THIS FOLDER 
#               UNLESS YOU KNOW EXACTLY WHAT YOU ARE DOING.
#========================================================================

CND_FILES:= $(wildcard $(cnd)/*.f)
CND_OBJTS:= $(CND_FILES:.f=.o)

#Compilation command follows
$(cnd)/%.o: $(cnd)/%.f
	$(F90) $(F90FLAGS) $(HE2FLAGS) -c -o $@ $<
 
#========================================================================
#            Free wake: source files and objects
# Preprocessing need extensions to be .F and .F90 (not .f and .f90)
#========================================================================

#fixed-format source files (MFW)
WKE_FILES:= $(wildcard $(wke)/*.F)
WKOBJECTS:= $(WKE_FILES:.F=.o)

WKE2FILES = $(wildcard $(wke)/*.F90)
WKOBJCTS2:= $(WKE2FILES:.F90=.o)

#Compilation commands follow for wake object files
$(wke)/%.o: $(wke)/%.F
	$(F90) $(F90FLAGS) $(WK_FLAGS) $(HE2FLAGS) -c -o $@ $<

$(wke)/%.o: $(wke)/%.F90
	$(F90) $(F90FLAGS) $(WK_FLAGS) $(HE_FLAGS) -c -o $@ $<

#========================================================================
#           Vehicle dynamics : source files and objects
#========================================================================

F90_FILES:= $(wildcard $(src)/*.F90)
OBJ_FILES:= $(F90_FILES:.F90=.o)

#Command to create object(.o) files from source files
$(src)/%.o: $(src)/%.F90
	$(F90) $(F90FLAGS) $(HE_FLAGS) $(WK_FLAGS) -c -o $@ $< 

#========================================================================
#                       CFD-CSD coupling files
#========================================================================

CPL_FILES:= $(wildcard $(cpl)/*.F90)
CPL_OBJTS:= $(CPL_FILES:.F90=.o)

#Command to create object(.o) files from source files
$(cpl)/%.o: $(cpl)/%.F90
	$(F90) $(F90FLAGS) -c -o $@ $< 

#========================================================================
#          Rotor Blade Dynamics : source files and objects
#========================================================================

BIT_FILES:= $(wildcard $(bit)/*.F90)
BIT_OBJTS:= $(BIT_FILES:.F90=.o)

#Command to create object(.o) files from source files
$(bit)/%.o: $(bit)/%.F90
	$(F90) $(F90FLAGS) $(HE_FLAGS) -c -o $@ $< 

#========================================================================
#        Generate object files needed for python interface 
#========================================================================

PY2_FILES:= $(wildcard $(py2)/*.F90)
PY2_OBJTS:= $(PY2_FILES:.F90=.o)

#Compilation command follows
$(py2)/%.o: $(py2)/%.F90
	$(F90) $(F90FLAGS) $(HE_FLAGS) -c -o $@ $<

#========================================================================
#                 An easy way to refer to all objects
#========================================================================

ALLOBJ:=  $(PY1_OBJTS) $(BSC_OBJTS) $(OBJ_FILES) $(CND_OBJTS) $(OBJ2FILES) \
          $(CPL_OBJTS) $(BIT_OBJTS) $(WKOBJECTS) $(WKOBJCTS2) $(PY2_OBJTS)

#========================================================================
#Creating the executables: list them out first..
#========================================================================

#========================================================================
#default: when you type "make", it will build these executables
#========================================================================

all:prasadum libsrc.a
default: prasadum

#========================================================================
#First executable
#========================================================================

prasadum: $(ALLOBJ)
	$(F90) $(F90FLAGS) $(ALLOBJ) -o prasadum

#========================================================================
# library file with all dependencies
#========================================================================

libsrc.a: $(ALLOBJ)
	$(AR) $(AROPT) $@ $(ALLOBJ)

#========================================================================
# shortcuts to remove objects and modules (when changing mods/compilers)
#========================================================================
clean_basic:
	rm -f prasadum $(bsc)/*.o $(py1)/*.o $(py2)/*.o $(cnd)/hybrd.o
clean_heli:
	rm -f prasadum $(src)/*.o $(cnd)/hybrd.o
clean_canned:
	rm -f $(cnd)/*.o $(cnd)/*.*~
clean_cable:
	rm -f $(ALLOB2) $(PROGRAM2)
clean_wake:
	rm -f $(wke)/*.o $(wke)/*.*~
clean:
	rm -f prasadum $(moddir)/*.mod $(ALLOBJ)
clean_mods:
	rm -f $(moddir)/*.mod 
chain_unlink:
	rm -f prasadum libsrc.a
