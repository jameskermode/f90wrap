  subroutine wrap(opt, def)
    implicit none

    integer :: def
    integer, optional :: opt

    print *, 'present(opt) = ', present(opt)
    print *, 'def = ', def

    if (present(opt)) print *, 'opt = ', opt

  end subroutine
