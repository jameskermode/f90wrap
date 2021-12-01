program test_circle
  use ClassCircle
  block

    ! Use block so destructor fires
    type(Circle) :: c
    double precision :: x

    c = Circle(5.0)
    x = c%get_area()
    x = c%get_radius()
    call c%print()

    c =  Circle('test', 3.0)
    x = c%get_area()
    x = c%get_radius()
    call c%print('tag')

  end block
end program
