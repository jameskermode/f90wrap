module subroutine_mod

    implicit none

contains

    subroutine routine_with_simple_args(a, b, c, d)
      integer, intent(in) :: a, b
      integer, intent(out) :: c, d

      c = a + b
      d = a * b

    end subroutine routine_with_simple_args

    subroutine routine_with_multiline_args(&
         & a, b, c, d)
      integer, intent(in) :: a, b
      integer, intent(out) :: c, d

      c = a + b
      d = a * b

    end subroutine routine_with_multiline_args

    subroutine routine_with_commented_args(&
         ! Some arbitrary comment...
         & a, b, &
         ! ... to highlight Fortran line continuation.
         & c, d)
      integer, intent(in) :: a, b
      integer, intent(out) :: c, d

      c = a + b
      d = a * b

    end subroutine routine_with_commented_args

    subroutine routine_with_more_commented_args(&
       ! Some additional commenting...
         & a, b, &
       ! ... to highlight the true ...
       ! ... beauty of the FORTRAN language.
         & c, d)
      integer, intent(in) :: a, b
      integer, intent(out) :: c, d

      c = a + b
      d = a * b

    end subroutine routine_with_more_commented_args

end module subroutine_mod
