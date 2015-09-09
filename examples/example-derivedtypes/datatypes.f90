! ==============================================================================
module datatypes_allocatable

use parameters, only: idp, isp

implicit none

! Why do the alloc_arrays work here and not in the datatypes module??
type alloc_arrays
    REAL(idp), DIMENSION(:,:), ALLOCATABLE :: chi
    REAL(idp), DIMENSION(:,:), ALLOCATABLE :: psi
    INTEGER(4) chi_shape(2)
    INTEGER(4) psi_shape(2)
end type alloc_arrays

contains

subroutine init_alloc_arrays(dertype, m, n)
    type(alloc_arrays), INTENT(inout) :: dertype
    INTEGER(4), INTENT(in) :: m, n
    allocate(dertype%chi(m,n))
    allocate(dertype%psi(m,n))
end subroutine init_alloc_arrays

subroutine destroy_alloc_arrays(dertype)
    type(alloc_arrays), INTENT(inout) :: dertype
    if (allocated(dertype%chi)) deallocate(dertype%chi)
    if (allocated(dertype%psi)) deallocate(dertype%psi)
end subroutine destroy_alloc_arrays

end module datatypes_allocatable
! ==============================================================================

! ==============================================================================
module datatypes

use parameters, only: idp, isp
use datatypes_allocatable, only: alloc_arrays

implicit none
!private
!public :: t_mass_data, t_element_data

! NB: target attribute needed to allow derived type access from Python.
! for gfortran, the mockdtderivetype use the following pre-processor flag:
! #define type(x) TYPE(x), target
! however, for my Linux-64bit installation, both gfortran (v5.2.0) and
! ifort (v15.0.3 20150407) did work fine withouth using the TARGET attribute

type different_types
    LOGICAL alpha
    INTEGER(4) beta
    REAL(idp) delta
end type different_types

! derived type with multiple arrays of vayring types, but fixed shape
type fixed_shape_arrays
    INTEGER(4) eta(10,4)
    REAL(isp) theta(10,4)
    REAL(idp) iota(10,4)
end type fixed_shape_arrays

type nested
    type(different_types) mu
    type(fixed_shape_arrays) nu
end type nested

! pointer arrays in a derived type are not accessible from Python
type pointer_arrays
    REAL(idp), DIMENSION(:,:), POINTER :: chi
    REAL(idp), DIMENSION(:,:), POINTER :: psi
    INTEGER(4) chi_shape(2)
    INTEGER(4) psi_shape(2)
end type pointer_arrays

! why is this not working here??
type alloc_arrays_2
    REAL(idp), DIMENSION(:,:), ALLOCATABLE :: chi
    REAL(idp), DIMENSION(:,:), ALLOCATABLE :: psi
    INTEGER(4) chi_shape(2)
    INTEGER(4) psi_shape(2)
end type alloc_arrays_2

type array_nested
    type(different_types), DIMENSION(:), ALLOCATABLE :: xi
    type(fixed_shape_arrays), DIMENSION(:), ALLOCATABLE :: omicron
    type(alloc_arrays), DIMENSION(:), ALLOCATABLE :: pi
end type array_nested

contains

subroutine init_array_nested(dertype, size)
    type(array_nested), INTENT(inout) :: dertype
    INTEGER(4), INTENT(in) :: size
    allocate(dertype%xi(size))
    allocate(dertype%omicron(size))
    allocate(dertype%pi(size))
end subroutine init_array_nested

subroutine destroy_array_nested(dertype)
    type(array_nested), INTENT(inout) :: dertype
    if (allocated(dertype%xi)) deallocate(dertype%xi)
    if (allocated(dertype%omicron)) deallocate(dertype%omicron)
    if (allocated(dertype%pi)) deallocate(dertype%pi)
end subroutine destroy_array_nested

end module datatypes
! ==============================================================================

