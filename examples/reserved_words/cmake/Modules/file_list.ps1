#Script for Powershell to build list of applicable file names (uses string manipulation)
$fwrap_files = gci f90wrap_*.f90 | select name | out-string -stream
$f90src=[string]${fwrap_files}.Replace('-',' ').replace('Name',' ').Trim()
echo $f90src | Out-File -filepath final_part2.txt