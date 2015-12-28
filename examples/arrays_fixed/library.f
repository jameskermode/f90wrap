      module library

      use parameters, only: idp, isp
      implicit none
      private
      public :: do_array_stuff, only_manipulate, return_array

      contains

      subroutine do_array_stuff(n,
     &                          x,y,br,co)
c-----------------------------------------------------------------------
      INTEGER, INTENT(in) :: n
      REAL(kind=idp), INTENT(in) :: x(n)
      REAL(kind=idp), INTENT(in) :: y(n)
      REAL(kind=idp), INTENT(out) :: br(n)
      REAL(kind=idp), INTENT(out) :: co(4,n)

      INTEGER :: i,j
      DO j=1,n
          br(j) =  x(j) / ( y(j) + 1.0_idp )
          DO i=1,4
              co(i,j) = x(j)*y(j) + x(j)
          ENDDO
      ENDDO
      end subroutine do_array_stuff

      subroutine only_manipulate(n,array)
      INTEGER, INTENT(in) :: n
      REAL(kind=idp), INTENT(inout) :: array(4,n)

      INTEGER :: i,j
      DO j=1,n
          DO i=1,4
              array(i,j) = array(i,j)*array(i,j)
          ENDDO
      ENDDO
      end subroutine only_manipulate

      subroutine return_array(m, n, output)
      INTEGER, INTENT(in) :: m, n
      INTEGER, INTENT(out) :: output(m,n)
      INTEGER :: i,j
      DO i=1,m
          DO j=1,n
              output(i,j) = i*j + j
          ENDDO
      ENDDO
      end subroutine return_array

      end module library

