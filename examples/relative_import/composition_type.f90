module m_composition
    use m_base_type, only: t_base_type
    implicit none
    private

    type, public :: t_composition
       type(t_base_type) :: member
    end type t_composition

  contains

end module m_composition
