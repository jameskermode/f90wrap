subroutine f90wrap_do_stuff(factor, out)
    
    ! BEGIN write_uses_lines
    use use_a_type
    ! END write_uses_lines
    
    implicit none
    
    ! BEGIN write_type_lines
    ! END write_type_lines
    
    ! BEGIN write_arg_decl_lines
    real(8), intent(in) :: factor
    real(8), intent(out) :: out
    ! END write_arg_decl_lines
    
    ! BEGIN write_transfer_in_lines
    ! END write_transfer_in_lines
    
    ! BEGIN write_init_lines
    ! END write_init_lines
    
    ! BEGIN write_call_lines
    call do_stuff(factor=factor, out=out)
    ! END write_call_lines
    
    ! BEGIN write_transfer_out_lines
    ! END write_transfer_out_lines
    
    ! BEGIN write_finalise_lines
    ! END write_finalise_lines
    
end subroutine f90wrap_do_stuff


