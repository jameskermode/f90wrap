module library

    use parameters, only: idp, isp
    implicit none

contains

    function return_value_func(val_in) result(val_out)
        INTEGER(4) val_in, val_out
        val_out = val_in + 10
    end function return_value_func

    subroutine return_value_sub(val_in, val_out)
        INTEGER(4), INTENT(in) :: val_in
        INTEGER(4), INTENT(out) :: val_out
        val_out = val_in + 10
    end subroutine return_value_sub

    function return_a_dt_func() result(dt)
        use datatypes, only: different_types
        type(different_types) :: dt
        dt%alpha = .true.
        dt%beta = 666
        dt%delta = 666.666_idp
    end function return_a_dt_func

    ! note that all memory allocation can now happen in Python
    subroutine do_array_stuff(n,x,y,br,co)
        INTEGER, INTENT(in) :: n
        REAL(kind=idp), INTENT(in) :: x(n)
        REAL(kind=idp), INTENT(in) :: y(n)
        REAL(kind=idp), INTENT(out) :: br(n)
        REAL(kind=idp), INTENT(out) :: co(4,n)
        INTEGER :: i,j
        
        DO j=1,n
            br(j) =  x(j) / ( y(j) + 1.0_idp )
            DO i=1,4
                co(i,j) = x(j)*y(j) + x(j)
            ENDDO
        ENDDO
    end subroutine do_array_stuff

    subroutine only_manipulate(n,array)
        INTEGER, INTENT(in) :: n
        REAL(kind=idp), INTENT(inout) :: array(4,n)
        INTEGER :: i,j

        DO j=1,n
            DO i=1,4
                array(i,j) = array(i,j)*array(i,j)
            ENDDO
        ENDDO
    end subroutine only_manipulate

    subroutine set_derived_type(dt, dt_beta, dt_delta)
        use datatypes, only: different_types
        INTEGER(4), INTENT(in) :: dt_beta
        REAL(kind=idp), INTENT(in) :: dt_delta
        TYPE(different_types), INTENT(out) :: dt

        dt%beta = dt_beta
        dt%delta = dt_delta
    end subroutine set_derived_type

    subroutine modify_derived_types(dt1, dt2, dt3)
        use datatypes, only: different_types
        TYPE(different_types), INTENT(inout) :: dt1, dt2, dt3
        
        dt1%beta = dt1%beta + 10
        dt1%delta = dt1%delta + 10.0_idp

        dt2%beta = dt2%beta + 20
        dt2%delta = dt2%delta + 20.0_idp

        dt3%beta = dt3%beta + 30
        dt3%delta = dt3%delta + 30.0_idp
    end subroutine modify_derived_types

    subroutine modify_dertype_fixed_shape_arrays(dertype)
        use datatypes, only: fixed_shape_arrays
        TYPE(fixed_shape_arrays), INTENT(out) :: dertype
        INTEGER :: i,j
        dertype%eta(:,:) = 10
        dertype%theta(:,:) = 2.0_isp
        dertype%iota(:,:) = 100.0_idp
    end subroutine modify_dertype_fixed_shape_arrays

    subroutine return_dertype_pointer_arrays(m, n, dertype)
        use datatypes, only: pointer_arrays
        INTEGER, INTENT(in) :: m, n
        TYPE(pointer_arrays), INTENT(out) :: dertype
        INTEGER :: i, j

        ALLOCATE(dertype%chi(m,n))
        dertype%chi(:,:) = 100.0_idp
        dertype%chi(m-2,n-1) = -10.0_idp
    end subroutine return_dertype_pointer_arrays

!    subroutine modify_dertype_pointer_arrays(dertype)
!        use datatypes, only: pointer_arrays
!        TYPE(pointer_arrays), INTENT(inout) :: dertype
!        INTEGER :: i, j, m, n
!        m = dertype%chi_shape(1)
!        n = dertype%chi_shape(2)
!        dertype%chi(:,:) = dertype%chi(:,:)*dertype%chi(:,:)
!        dertype%chi(m-2, n-1) = -9.0_idp
!    end subroutine modify_dertype_pointer_arrays

   subroutine return_dertype_alloc_arrays(m, n, dertype)
        use datatypes_allocatable, only: alloc_arrays
        INTEGER, INTENT(in) :: m, n
        TYPE(alloc_arrays), INTENT(out) :: dertype
        INTEGER :: i,j

        ALLOCATE(dertype%chi(m,n))
        dertype%chi(:,:) = 10.0_idp
        dertype%chi(m-2,n-1) = -1.0_idp
    end subroutine return_dertype_alloc_arrays

    subroutine modify_dertype_alloc_arrays(dertype)
        use datatypes_allocatable, only: alloc_arrays
        TYPE(alloc_arrays), INTENT(inout) :: dertype
        INTEGER :: i, j, m, n
        m = dertype%chi_shape(1)
        n = dertype%chi_shape(2)
        dertype%chi(:,:) = dertype%chi(:,:)*dertype%chi(:,:)
        dertype%chi(m-2, n-1) = -9.0_idp
    end subroutine modify_dertype_alloc_arrays

    ! Python bindings for this subroutine doesn't work either because of a
    ! pointer/allocator argument
    subroutine return_array_nested(dt_array)

        use datatypes, only: array_nested, different_types, fixed_shape_arrays
        use datatypes_allocatable, only: alloc_arrays

        type(array_nested), INTENT(out), DIMENSION(:), ALLOCATABLE :: dt_array
        type(array_nested) :: dt, dt_tmp
        type(different_types) :: xi
        type(fixed_shape_arrays) :: omicron
        type(alloc_arrays) :: pi
        INTEGER :: i

        ! type(different_types), DIMENSION(:), ALLOCATABLE :: xi
        ! type(fixed_shape_arrays), DIMENSION(:), ALLOCATABLE :: omicron
        ! type(alloc_arrays), DIMENSION(:), ALLOCATABLE :: pi

        if (allocated(dt_array)) deallocate(dt_array)
        allocate(dt_array(3))

        do i=1,3
            dt = dt_array(i)
            dt%xi%beta = 2*i
            dt%xi%delta = 2*i
        end do

    end subroutine return_array_nested

!    subroutine modify_dertype_pointer_arrays(dertype)
!        use datatypes, only: pointer_arrays
!        TYPE(pointer_arrays), INTENT(inout), POINTER :: dertype
!!        REAL(idp), DIMENSION(:,:), ALLOCATABLE :: kappa
!!        REAL(idp), DIMENSION(:,:), POINTER :: lambda
!        INTEGER :: i,j,m,n
!        REAL(idp) :: tmp_idp

!        n = 12
!        m = 4
!        ! create the new vector and give it some values
!!        if (allocated(kappa)   deallocate(kappa)
!        ALLOCATE(dertype%kappa(n,m))
!        DO i=1,n
!            DO j=1,m
!                ! for the sake of the example, convert to a float
!                tmp_idp = TRANSFER(i*j, tmp_idp)
!                dertype%kappa(i,j) = tmp_idp
!            ENDDO
!        ENDDO
!        dertype%lambda => dertype%kappa
!    end subroutine modify_dertype_pointer_arrays

end module library

