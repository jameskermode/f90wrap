module leveltwomod
    implicit none
    !% Define a type which is referenced by the primary used type, to make
    !% sure it is also wrapped (and we can access its elements).
    type leveltwo
        real(8) :: rl
    end type leveltwo

end module leveltwomod
