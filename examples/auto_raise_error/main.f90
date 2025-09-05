module m_error
  implicit none
  private

  public :: auto_raise,auto_no_raise
  public :: auto_raise_optional,auto_no_raise_optional
  public :: no_error_var
  public :: str_input

contains

  subroutine str_input(keyword)

    implicit none
    character(len=*),   optional, intent(in)    :: keyword

  end subroutine

  subroutine auto_raise(ierr,errmsg)

    implicit none
    integer,          intent(out) :: ierr
    character(len=*), intent(out) :: errmsg

    ierr=1
    write(errmsg,'(a)') 'auto raise error'
    return

  end subroutine

  subroutine auto_raise_optional(ierr,errmsg)

    implicit none
    integer,optional,          intent(out) :: ierr
    character(len=*),optional, intent(out) :: errmsg

    ierr=1
    write(errmsg,'(a)') 'auto raise error optional'
    return

  end subroutine

  subroutine auto_no_raise(ierr,errmsg)

    implicit none
    integer,          intent(out) :: ierr
    character(len=*), intent(out) :: errmsg

    ierr=0
    write(errmsg,'(a)') ''
    return

  end subroutine

  subroutine auto_no_raise_optional(ierr,errmsg)

    implicit none
    integer,optional,          intent(out) :: ierr
    character(len=*),optional, intent(out) :: errmsg

    ierr=0
    write(errmsg,'(a)') ''
    return

  end subroutine

  subroutine no_error_var(a_num,a_string)

    implicit none
    integer,          intent(out) :: a_num
    character(len=*), intent(out) :: a_string

    a_num=1
    write(a_string,'(a)') 'a string'
    return

  end subroutine



end module m_error
