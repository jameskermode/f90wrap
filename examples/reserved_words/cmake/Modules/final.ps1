#======================================================================================
#Powershell script to read 3 parts of final compilation command from files, assemble into one string and execute it
#======================================================================================
#
$part1=[IO.File]::ReadAllText([String]$pwd+"\final_part1.txt")
#
#======================================================================================
# get list of files matching the pattern f90wrap_*.f90
#======================================================================================
#
$fwrap_files = gci f90wrap_*.f90 | select name | out-string -stream
$f90src=[string]${fwrap_files}.Replace('-',' ').replace('Name',' ').Trim()
$part2=$f90src
#
#======================================================================================
# other compilation options
#======================================================================================
#
$part3=[IO.File]::ReadAllText([String]$pwd+"\final_part3.txt")
#
#======================================================================================
# Assemble final command
#======================================================================================
#
$com=$part1+' '+$part2+' '+$part3
echo $com
invoke-expression $com