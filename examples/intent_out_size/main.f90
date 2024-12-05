module m_intent_out

  implicit none
  public

contains

  subroutine interpolation(n1,n2,a1,a2,output)
    !
    integer,                   intent(in)    :: n1,n2
    real,dimension(n1,n2),     intent(in)    :: a1,a2
    real,dimension(n1,n2),     intent(out)   :: output

    integer :: i,j

    do j=1,n2
      do i=1,n1
         output(i,j)=(a1(i,j)+a2(i,j))/2
      enddo
    enddo

  end subroutine interpolation

end module m_intent_out


