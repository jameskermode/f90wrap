module myclass_base
implicit none

type, abstract :: myclass_t
contains
    procedure(get_value_i), deferred :: get_value
end type myclass_t

abstract interface
    subroutine get_value_i(self, value)
        import myclass_t
        class(myclass_t), intent(in) :: self
        real, intent(out) :: value
    end subroutine get_value_i
end interface

end module myclass_base
