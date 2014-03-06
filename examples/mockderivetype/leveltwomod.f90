module leveltwomod
  
    implicit none

    ! In reality, the several second-level types involved
    ! contain reals, ints, real vectors (not allocatable) and logicals.
    ! But I'm just going to have a single real.
    type leveltwo
        real(8) :: rl
    end type leveltwo

  contains

    subroutine leveltwo_initialise(this, rl)
      type(leveltwo), intent(out) :: this
      real(8), intent(in) :: rl

      this%rl = rl
      
    end subroutine leveltwo_initialise

    subroutine leveltwo_finalise(this)
      type(leveltwo), intent(inout) :: this

      this%rl = 0.0
    end subroutine leveltwo_finalise

    subroutine leveltwo_print(this)
      type(leveltwo), intent(in) :: this

      write(*,*) 'rl=', this%rl

    end subroutine leveltwo_print

end module leveltwomod
