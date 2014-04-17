module define_a_type
    use leveltwomod
    type atype
        logical     ::  bool
        integer     ::  integ
        real(8)    ::  rl
        real(8)    :: vec(10)
        type(leveltwo) :: dtype
    end type atype

    type unused_type
        real(8) :: rl
    end type unused_type
contains

end module

