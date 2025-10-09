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

  public :: construct_circle
  public :: construct_circle_with_more_args,construct_circle_more_doc
  public :: incomplete_doc_sub
  public :: no_direction,doc_inside
  public :: output_1,function_2
  public :: ierr_errmsg
  public :: details_doc,details_with_parenthesis
  public :: multiline_details,empty_lines_details,long_line_brief


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
    !! \brief Initialize circle with more args
    !! \param[in,out] circle      t_circle to initialize
    !! \param[in]     radius1,radius2      radius of the circle
    !<
  subroutine construct_circle_with_more_args(circle,radius1,radius2)
    type(t_circle) :: circle
    real, intent(in) :: radius1,radius2
    circle%radius = radius1 + radius2
  end subroutine construct_circle_with_more_args

    !===========================================================================
    !>
    !! \brief Initialize circle with more doc
    !! \author test_author
    !! \copyright test_copyright
    !! \param[in,out] circle      t_circle to initialize
    !! \param[in]     radius      radius of the circle
    !<
  subroutine construct_circle_more_doc(circle,radius)
    type(t_circle) :: circle
    real, intent(in) :: radius
    circle%radius = radius
  end subroutine construct_circle_more_doc

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
    !! \brief subroutine is able to raise error
    !! \param[out]     output       this is 1
    !! \param[out]     ierr         error code
    !! \param[out]     errmsg       error message
    !<
  subroutine ierr_errmsg(output,ierr,errmsg)
    real,                 intent(out) :: output
    integer,              intent(out) :: ierr
    character(len=*),     intent(out) :: errmsg
    output = 1
  end subroutine ierr_errmsg

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

    !===========================================================================
    !>
    !! \brief Initialize circle
    !! \details Those are very informative details (with parenthesis)
    !! \param[in,out] circle      t_circle to initialize
    !! \param[in]     radius      radius of the circle
    !<
  subroutine details_with_parenthesis(circle,radius)
    type(t_circle) :: circle
    real, intent(in) :: radius
  end subroutine details_with_parenthesis

    !===========================================================================
    !>
    !! \brief Initialize circle
    !! \details First details line
    !!        Second details line
    !! \param[in,out] circle      t_circle to initialize
    !! \param[in]     radius      radius of the circle
    !<
  subroutine multiline_details(circle,radius)
    type(t_circle) :: circle
    real, intent(in) :: radius
  end subroutine multiline_details

    !===========================================================================
    !>
    !! \brief Initialize circle
    !! \details First details line
    !!
    !!        Second details line after a empty line
    !! \param[in,out] circle      t_circle to initialize
    !! \param[in]     radius      radius of the circle
    !<
  subroutine empty_lines_details(circle,radius)
    type(t_circle) :: circle
    real, intent(in) :: radius
  end subroutine empty_lines_details

    !===========================================================================
    !>
    !! \brief This is a very long brief that takes up a lot of space and contains lots of information, it should probably be wrapped to the next line, but we will continue regardless
    !! \details Those are very informative details
    !! \param[in,out] circle      t_circle to initialize
    !! \param[in]     radius      radius of the circle
    !<
  subroutine long_line_brief(circle,radius)
    type(t_circle) :: circle
    real, intent(in) :: radius
  end subroutine long_line_brief

end module m_circle
