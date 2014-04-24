!% This module is used by the top-level module and so should be wrapped.
!% However, the subroutine will NOT be wrapped if the subroutine from
!% use_a_type is given explicitly.
module define_a_type
    use leveltwomod

    !% This type will be wrapped as it is used.
    type atype
        logical     ::  bool
        integer     ::  integ
        real(8)    ::  rl
        real(8)    :: vec(10)
        type(leveltwo) :: dtype
    end type atype

    real(8) :: a_set_real = 4.d0
    logical :: a_set_bool = .true.

    !% This type will also be wrapped, but we may not want it to be??
    type unused_type
        real(8) :: rl = 3.d0
    end type

    contains

    subroutine use_set_vars()
        if(a_set_bool)then
            call top_level(a_set_real,a_set_real)
            a_set_bool = .false.
        endif
    end subroutine use_set_vars
end module

!% Example of a top-level subroutine. This is used in the above module, but
!% at the moment cannot be explicitly wrapped. We may want it to be though.
subroutine top_level(input,out)
    real(8), intent(in) :: input ! FIXME renamed arg 'in' -> 'input' to prevent clash with Python reserved word
    real(8), intent(out) :: out
    out = 85.d0*in
end subroutine top_level

!% The following module is an example of something that is not wrapped because
!% it is not used by any of the primary modules. We may want to exclude it as
!% it contains a complex type which is difficult to wrap and we never need to
!% explicitly call it from python anyway.
module not_wrapped
    use leveltwomod
    real(8) :: a_real
    type horrible_type
        real(8), allocatable :: x(:,:,:)
    end type
end module
