
module caller
    use cback
    implicit none
    
    public

contains

subroutine test_write_msg()
    call write_message("from test_write_msg")
    return
end

subroutine test_write_msg_2()
    call write_message("from test_write_msg_2")
    return
end

! character(len=20) function test_return_msg()
!     test_return_msg = return_message("from test_return_msg")
! end

! character(len=20) function test_return_msg_2()
!     test_return_msg_2 = return_message("from test_return_msg_2")
! end

end module