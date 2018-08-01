
import subroutine_mod

mod = subroutine_mod.Subroutine_Mod

c, d = mod.routine_with_simple_args(2, 3)
assert c == 5 and d == 6

c, d = mod.routine_with_multiline_args(2, 3)
assert c == 5 and d == 6

c, d = mod.routine_with_commented_args(2, 3)
assert c == 5 and d == 6

c, d = mod.routine_with_more_commented_args(2, 3)
assert c == 5 and d == 6
