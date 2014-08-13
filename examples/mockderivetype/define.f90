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

    !% This type will also be wrapped
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

    function return_a_type_func() result(a)
      type(atype) :: a

      a%bool = .true.
      a%integ = 42

    end function return_a_type_func

    subroutine return_a_type_sub(a)
      type(atype), intent(out) :: a

      a%bool = .true.
      a%integ = 42

    end subroutine return_a_type_sub

end module

!% Example of a top-level subroutine.
subroutine top_level(in, out)
    real(8), intent(in) :: in
    real(8), intent(out) :: out
    out = 85.d0*in
end subroutine top_level

!% Another module, defining a horrible type
module horrible
    use leveltwomod
    real(8) :: a_real
    type horrible_type
        real(8), allocatable :: x(:,:,:)
    end type
end module

