module define_a_type
    use leveltwomod
    type atype
        logical     ::  bool
        integer     ::  integ
        real(8)    ::  rl
        real(8)    :: vec(10)
        type(leveltwo) :: dtype
    end type atype

contains

!  subroutine atype_initialise(this)
!    type(atype), intent(out) :: this

!    write (*,*) 'initialising atype'
    
!  end subroutine atype_initialise

!  subroutine atype_finalise(this)
!    type(atype), intent(inout) :: this

!    write (*,*) 'finalising atype'

!  end subroutine atype_finalise

end module

