subroutine f90wrap_leveltwo_initialise(this, rl)
    
    ! BEGIN write_uses_lines
    use leveltwomod
    ! END write_uses_lines
    
    implicit none
    
    ! BEGIN write_type_lines
    type leveltwo_ptr_type
        type(leveltwo), pointer :: p => NULL()
    end type leveltwo_ptr_type
    ! END write_type_lines
    
    ! BEGIN write_arg_decl_lines
    type(leveltwo_ptr_type) :: this_ptr
    integer, intent(out), dimension(4) :: this
    real(8), intent(in) :: rl
    ! END write_arg_decl_lines
    
    ! BEGIN write_transfer_in_lines
    ! END write_transfer_in_lines
    
    ! BEGIN write_init_lines
    allocate(this_ptr%p)
    ! END write_init_lines
    
    ! BEGIN write_call_lines
    call leveltwo_initialise(this=this_ptr%p, rl=rl)
    ! END write_call_lines
    
    ! BEGIN write_transfer_out_lines
    this = transfer(this_ptr, this)
    ! END write_transfer_out_lines
    
    ! BEGIN write_finalise_lines
    ! END write_finalise_lines
    
end subroutine f90wrap_leveltwo_initialise


subroutine f90wrap_leveltwo_finalise(this)
    
    ! BEGIN write_uses_lines
    use leveltwomod
    ! END write_uses_lines
    
    implicit none
    
    ! BEGIN write_type_lines
    type leveltwo_ptr_type
        type(leveltwo), pointer :: p => NULL()
    end type leveltwo_ptr_type
    ! END write_type_lines
    
    ! BEGIN write_arg_decl_lines
    type(leveltwo_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    ! END write_arg_decl_lines
    
    ! BEGIN write_transfer_in_lines
    this_ptr = transfer(this, this_ptr)
    ! END write_transfer_in_lines
    
    ! BEGIN write_init_lines
    ! END write_init_lines
    
    ! BEGIN write_call_lines
    call leveltwo_finalise(this=this_ptr%p)
    ! END write_call_lines
    
    ! BEGIN write_transfer_out_lines
    ! END write_transfer_out_lines
    
    ! BEGIN write_finalise_lines
    deallocate(this_ptr%p)
    ! END write_finalise_lines
    
end subroutine f90wrap_leveltwo_finalise


subroutine f90wrap_leveltwo_print(this)
    
    ! BEGIN write_uses_lines
    use leveltwomod
    ! END write_uses_lines
    
    implicit none
    
    ! BEGIN write_type_lines
    type leveltwo_ptr_type
        type(leveltwo), pointer :: p => NULL()
    end type leveltwo_ptr_type
    ! END write_type_lines
    
    ! BEGIN write_arg_decl_lines
    type(leveltwo_ptr_type) :: this_ptr
    integer, intent(in), dimension(4) :: this
    ! END write_arg_decl_lines
    
    ! BEGIN write_transfer_in_lines
    this_ptr = transfer(this, this_ptr)
    ! END write_transfer_in_lines
    
    ! BEGIN write_init_lines
    ! END write_init_lines
    
    ! BEGIN write_call_lines
    call leveltwo_print(this=this_ptr%p)
    ! END write_call_lines
    
    ! BEGIN write_transfer_out_lines
    ! END write_transfer_out_lines
    
    ! BEGIN write_finalise_lines
    ! END write_finalise_lines
    
end subroutine f90wrap_leveltwo_print


