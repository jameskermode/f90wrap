! Minimal reproducer for direct-c code generation bug
! Based on patterns from gvec's t_sfl_boozer that cause C compilation errors

module types
    implicit none
    private

    integer, parameter :: dp = kind(1.0d0)

    ! Simple helper type that will be used as allocatable component
    type, public :: t_helper
        integer :: n
        real(dp), allocatable :: data(:)
    contains
        procedure :: init => helper_init
        procedure :: free => helper_free
    end type t_helper

    ! Main type with allocatable arrays and allocatable derived type component
    ! This pattern triggers the direct-c code generation bug
    type, public :: t_container
        logical :: initialized = .false.
        integer :: size
        ! Allocatable arrays
        real(dp), allocatable :: values(:)
        real(dp), allocatable :: matrix(:,:)
        ! Allocatable derived type component
        type(t_helper), allocatable :: helper
    contains
        procedure :: init => container_init
        procedure :: free => container_free
        procedure :: get_values => container_get_values
    end type t_container

    public :: container_new

contains

    subroutine helper_init(self, n)
        class(t_helper), intent(inout) :: self
        integer, intent(in) :: n
        self%n = n
        allocate(self%data(n))
        self%data = 0.0_dp
    end subroutine helper_init

    subroutine helper_free(self)
        class(t_helper), intent(inout) :: self
        if (allocated(self%data)) deallocate(self%data)
        self%n = 0
    end subroutine helper_free

    subroutine container_new(c, n, m)
        type(t_container), allocatable, intent(inout) :: c
        integer, intent(in) :: n, m

        allocate(c)
        c%size = n
        allocate(c%values(n))
        allocate(c%matrix(n, m))
        allocate(c%helper)
        call c%helper%init(n)
        c%values = 1.0_dp
        c%matrix = 2.0_dp
        c%initialized = .true.
    end subroutine container_new

    subroutine container_init(self, n, m)
        class(t_container), intent(inout) :: self
        integer, intent(in) :: n, m

        self%size = n
        allocate(self%values(n))
        allocate(self%matrix(n, m))
        allocate(self%helper)
        call self%helper%init(n)
        self%values = 1.0_dp
        self%matrix = 2.0_dp
        self%initialized = .true.
    end subroutine container_init

    subroutine container_free(self)
        class(t_container), intent(inout) :: self

        if (allocated(self%values)) deallocate(self%values)
        if (allocated(self%matrix)) deallocate(self%matrix)
        if (allocated(self%helper)) then
            call self%helper%free()
            deallocate(self%helper)
        end if
        self%initialized = .false.
    end subroutine container_free

    subroutine container_get_values(self, out_values)
        class(t_container), intent(in) :: self
        real(dp), intent(out) :: out_values(:)

        out_values = self%values
    end subroutine container_get_values

end module types
