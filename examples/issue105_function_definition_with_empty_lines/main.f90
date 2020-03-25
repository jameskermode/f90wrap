module itestit

  implicit none

  private
  
  public :: testit1
  public :: testit2
  
contains

  subroutine testit1( &
       x &
       )

    implicit none

    real, dimension(:), intent(inout) :: x

    x = 1.*x

  end subroutine testit1

  subroutine testit2( &
       x &

       )

    implicit none

    real, dimension(:), intent(inout) :: x

    x = 2.*x

  end subroutine testit2

end module itestit
