  function(win32_pshell code)
    mktemp()
    ans(path)

    fwrite("${path}/script.ps1" "${code}")
    ans(script_file)
    win32_powershell(
      -NoLogo                   # no info output 
      -NonInteractive           # no interaction
      -ExecutionPolicy ByPass   # bypass execution policy 
     # -NoNewWindow              
      #-WindowStyle Hidden       # hide window
      -File "${script_file}"    # the file to execute
      ${ARGN}                   # add further args to command line
      )
    return_ans()
  endfunction()