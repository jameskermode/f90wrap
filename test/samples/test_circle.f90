program test_circle
  use ClassCircle
  block
    ! Use block so destructor fires
    type(Circle) :: c
    c = Circle(5.0)
    write(*,*) 'get_area:   ', c%get_area()
    write(*,*) 'get_radius: ', c%get_radius()
    call c%print()
  end block
end program
