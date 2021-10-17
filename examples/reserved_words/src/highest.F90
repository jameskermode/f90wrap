#define type(X) Type(X), target

module case
    
    implicit none 
    real(kind=4) :: module_var
    
end module

module highest_level
    use case
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
        Type(Size) :: Size!, inner_type_array(1)
        Type(oktype) :: Oktype
    end type

    type(oktype) :: oktype_tmp
    type(outer) :: outer_tmp
    type(Size) :: len
end module
