module KIMDispersion_Horton_module
    use KIMDispersionEquation_module
    implicit none

    type, extends(KIMDispersionEquation) :: KIMDispersion_Horton
        type(OptionsType) :: options
    contains
        procedure :: initialize => initialize_KIMDispersion_Horton
    end type KIMDispersion_Horton

contains

    subroutine initialize_KIMDispersion_Horton(this, options)
        class(KIMDispersion_Horton), intent(inout) :: this
        type(OptionsType), intent(in) :: options

        this%options = options
    end subroutine initialize_KIMDispersion_Horton

end module KIMDispersion_Horton_module
