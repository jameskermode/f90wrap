
module cback
    implicit none
    
    public

    contains
    
    subroutine write_message(msg)
        !f2py    intent(callback, hide) pyfunc_print
        external pyfunc_print
        character(len= *) :: msg
        call pyfunc_print(msg)
    end

    ! TODO cannot seem to make it work. Leads to "error: ‘PyStringObject’ undeclared"
    ! character(len=20) function return_message(msg)
    !     !f2py    intent(callback, hide) pyfunc_return
    !     external pyfunc_return
    !     character(len=20) pyfunc_return
    !     !f2py character(len=20) y,x
    !     !f2py y = py_func(x)
    !     character(len= *) :: msg
    !     character(len=20) :: returned_msg
    !     write(returned_msg,*)msg
    !     returned_msg = pyfunc_return(msg)
    !     return_message = returned_msg(1:20)
    ! end

end module