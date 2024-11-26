module myclass_factory

use myclass, only: myclass_t, create_count
implicit none

contains

function myclass_create(val) result(myobject)
    class(myclass_t), allocatable :: myobject
    real, intent(in) :: val

    allocate(myclass_t :: myobject)
    call myobject%set_val(val)
    create_count = create_count + 1

end function myclass_create

end module myclass_factory
