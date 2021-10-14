#define type(X) Type(X), target

module highest_level
    implicit none
    type :: Size
        sequence
        real(kind=8) :: test_double=1.2d2
        real(kind=4) :: test_single=0.1e2
        real         :: test_float=0.2e2

    end type 

    type :: oktype
        sequence
        real(kind=4) :: test_single=0.1e2
    end type 

    type :: outer
        sequence
        Type(Size) :: Size
        Type(oktype) :: Oktype
    end type

    type(Size) :: size_tmp
    type(oktype) :: oktype_tmp
    type(outer) :: outer_tmp
end module