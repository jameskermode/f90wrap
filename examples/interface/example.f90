!-*-f90-*-                                                                                                                                                                             
module class_example

  implicit none
  private
  public :: Example, return_example

  interface return_example
     module procedure return_example_first, return_example_second, return_example_third
  end interface return_example

  type Example
     integer :: first
     integer :: second
     integer :: third
  end type Example

  ! singleton                                                                                                                                                                          
  type(Example) :: this 

contains

!------------------------------------------------------------------------------------!                                                                                                 

  function return_example_first(first) result(instance)

    implicit none
    integer :: first
    type(Example) :: instance

    this%first = first
    instance = this

    return

  end function return_example_first

!------------------------------------------------------------------------------------!        

  function return_example_second(first,second) result(instance)                                                                                                                        

    implicit none
    integer :: first
    integer :: second
    type(Example) :: instance

    this%first = first
    this%second = second

    instance = this
    return

  end function return_example_second

!------------------------------------------------------------------------------------!                                                                                                 

  function return_example_third(first,second,third) result(instance)

    implicit none
    integer :: first
    integer :: second
    integer :: third
    type(Example) :: instance

    this%first = first
    this%second = second
    this%third = third

    instance = this
    return

  end function return_example_third

!------------------------------------------------------------------------------------!                                                                                                 
end module class_example