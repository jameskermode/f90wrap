module m_long_subroutine_name

    implicit none

    integer :: m_long_subroutine_name_integer

    type m_long_subroutine_name_type

        integer :: m_long_subroutine_name_type_integer
        integer, dimension(10) :: m_long_subroutine_name_type_integer_array

    end type m_long_subroutine_name_type

    type m_long_subroutine_name_type_2

        type(m_long_subroutine_name_type), dimension(10) :: m_long_subroutine_name_type_2_type_array

    end type m_long_subroutine_name_type_2

contains

    subroutine m_long_subroutine_name_subroutine()
    end subroutine m_long_subroutine_name_subroutine

end module m_long_subroutine_name