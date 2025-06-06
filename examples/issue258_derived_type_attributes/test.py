"""Test derived type as attributes of other derived types.

Check all combinations of type / class (polymorphic derived type) for the outer and intner derived types."""
import dta_tt
import dta_ct
import dta_tc
import dta_cc

for mod, o, i in [
    (dta_tt.dta_tt, "t", "t"),
    (dta_ct.dta_ct, "c", "t"),
    (dta_tc.dta_tc, "t", "c"),
    (dta_cc.dta_cc, "c", "c"),
]:
    print("Testing module:", mod)
    inner1 = mod.t_inner(1)
    assert inner1.value == 1
    inner2 = mod.t_inner(2)
    assert inner2.value == 2
    if i == "c":
        inner1.print()
        inner2.print()

    outer = mod.t_outer(10, inner1)
    assert outer.value == 10
    assert outer.inner.value == 1

    outer.inner = inner2
    assert outer.inner.value == 2
    if o == "c":
        outer.print()

