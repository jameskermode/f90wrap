! Module myclass defined in file myclass.f90

subroutine f90wrap_myclass_t__get__val(this, f90wrap_val)
    use myclass, only: myclass_t
    implicit none
    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    integer, intent(in)   :: this(2)
    type(myclass_t_ptr_type) :: this_ptr
    real, intent(out) :: f90wrap_val

    this_ptr = transfer(this, this_ptr)
    f90wrap_val = this_ptr%p%obj%val
end subroutine f90wrap_myclass_t__get__val

subroutine f90wrap_myclass_t__set__val(this, f90wrap_val)
    use myclass, only: myclass_t
    implicit none
    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    integer, intent(in)   :: this(2)
    type(myclass_t_ptr_type) :: this_ptr
    real, intent(in) :: f90wrap_val

    this_ptr = transfer(this, this_ptr)
    this_ptr%p%obj%val = f90wrap_val
end subroutine f90wrap_myclass_t__set__val

subroutine f90wrap_myclass__myclass_destroy__binding__myclass_t(self)
    use myclass, only: myclass_destroy, myclass_t
    implicit none

    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    type(myclass_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    self_ptr = transfer(self, self_ptr)
    call myclass_destroy(self=self_ptr%p%obj)
    deallocate(self_ptr%p)
end subroutine f90wrap_myclass__myclass_destroy__binding__myclass_t

subroutine f90wrap_myclass__myclass_t_initialise(this)
    use myclass, only: myclass_t
    implicit none

    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    type(myclass_t_ptr_type) :: this_ptr
    integer, intent(out), dimension(2) :: this
    allocate(this_ptr%p)
    this = transfer(this_ptr, this)
end subroutine f90wrap_myclass__myclass_t_initialise

subroutine f90wrap_myclass__get_val__binding__myclass_t(self, val)
    use myclass, only: myclass_t
    implicit none

    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    type(myclass_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    real, intent(out) :: val
    self_ptr = transfer(self, self_ptr)
    call self_ptr%p%obj%get_val(val=val)
end subroutine f90wrap_myclass__get_val__binding__myclass_t

subroutine f90wrap_myclass__set_val__binding__myclass_t(self, val)
    use myclass, only: myclass_t
    implicit none

    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    type(myclass_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    real, intent(in) :: val
    self_ptr = transfer(self, self_ptr)
    call self_ptr%p%obj%set_val(val=val)
end subroutine f90wrap_myclass__set_val__binding__myclass_t

subroutine f90wrap_myclass__myclass_get_val(self, val)
    use myclass, only: myclass_get_val, myclass_t
    implicit none

    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    type(myclass_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    real, intent(out) :: val
    self_ptr = transfer(self, self_ptr)
    call myclass_get_val(self=self_ptr%p%obj, val=val)
end subroutine f90wrap_myclass__myclass_get_val

subroutine f90wrap_myclass__myclass_set_val(self, val)
    use myclass, only: myclass_t, myclass_set_val
    implicit none

    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    type(myclass_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    real, intent(in) :: val
    self_ptr = transfer(self, self_ptr)
    call myclass_set_val(self=self_ptr%p%obj, val=val)
end subroutine f90wrap_myclass__myclass_set_val

subroutine f90wrap_myclass__myclass_destroy(self)
    use myclass, only: myclass_destroy, myclass_t
    implicit none

    type myclass_t_wrapper_type
        class(myclass_t), allocatable :: obj
    end type myclass_t_wrapper_type
    type myclass_t_ptr_type
        type(myclass_t_wrapper_type), pointer :: p => NULL()
    end type myclass_t_ptr_type
    type(myclass_t_ptr_type) :: self_ptr
    integer, intent(in), dimension(2) :: self
    self_ptr = transfer(self, self_ptr)
    call myclass_destroy(self=self_ptr%p%obj)
end subroutine f90wrap_myclass__myclass_destroy

! End of module myclass defined in file myclass.f90
