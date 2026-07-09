module widget_mod
    implicit none
    public

    type :: widget_t
        real :: x
        real :: scale
    end type widget_t

    interface widget_t
        module procedure widget_init
    end interface widget_t

contains

    function widget_init(x, scale) result(w)
        real, intent(in) :: x
        real, intent(in), optional :: scale
        type(widget_t) :: w
        w%x = x
        if (present(scale)) then
            w%scale = scale
        else
            w%scale = 1.0
        end if
    end function widget_init

end module widget_mod
