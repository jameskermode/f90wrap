!>
!! \file main.f90
!! \brief Test program docstring
!! \author test_author
!! \copyright test_copyright
!>
module m_circle
  implicit none
  private

  type, public :: t_circle
     real :: radius
  end type t_circle

  public :: construct_circle,incomplete_doc_sub
  public :: no_direction,doc_inside
  public :: output_1,function_2,details_doc

contains

    !===========================================================================
    !>
    !! \brief Initialize circle
    !! \param[in,out] circle      t_circle to initialize
    !! \param[in]     radius      radius of the circle
    !<
  subroutine construct_circle(circle,radius)
    type(t_circle) :: circle
    real, intent(in) :: radius
    circle%radius = radius
  end subroutine construct_circle

    !===========================================================================
    !>
    !! \brief Without direction
    !! \param circle      t_circle to initialize
    !! \param     radius      radius of the circle
    !<
  subroutine no_direction(circle,radius)
    type(t_circle) :: circle
    real, intent(in) :: radius
    circle%radius = radius
  end subroutine no_direction

    !===========================================================================
    !>
    !! \brief Incomplete doc
    !! \param[in]     radius      radius of the circle
    !<
  subroutine incomplete_doc_sub(circle,radius)
    type(t_circle) :: circle
    real, intent(in) :: radius
    circle%radius = radius
  end subroutine incomplete_doc_sub

  subroutine doc_inside(circle,radius)

    !===========================================================================
    !>
    !! \brief Doc inside
    !! \param[in,out] circle      t_circle to initialize
    !! \param[in]     radius      radius of the circle
    !<
    type(t_circle) :: circle
    real, intent(in) :: radius
    circle%radius = radius
  end subroutine doc_inside

    !===========================================================================
    !>
    !! \brief subroutine output_1 outputs 1
    !! \param[out]     output      this is 1
    !<
  subroutine output_1(output)
    real, intent(out) :: output
    output = 1
  end subroutine output_1

  !===========================================================================
  !>
  !! \brief this is a function
  !! \returns return value
  !! \param[in] input value
  !>
  integer function function_2(input)
    character(len=*),intent(in)    :: input
    function_2 = 2
  end function function_2

    !===========================================================================
    !>
    !! \brief Initialize circle
    !! \details Those are very informative details
    !! \param[in,out] circle      t_circle to initialize
    !! \param[in]     radius      radius of the circle
    !<
  subroutine details_doc(circle,radius)
    type(t_circle) :: circle
    real, intent(in) :: radius
  end subroutine details_doc

end module m_circle


