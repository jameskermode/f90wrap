module m_inheritance
    use m_base_type, only: t_base_type
    implicit none
    private

    type, public, extends(t_base_type) :: t_inheritance
       integer :: integer_number = 0
    end type t_inheritance

  contains

end module m_inheritance
