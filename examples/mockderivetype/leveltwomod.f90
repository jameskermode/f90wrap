module leveltwomod
  
    implicit none

    ! In reality, the several second-level types involved
    ! contain reals, ints, real vectors (not allocatable) and logicals.
    ! But I'm just going to have a single real.
    type leveltwo
        real(8) :: rl
    end type leveltwo

end module leveltwomod
