from mytype import mytype_mod

mine = mytype_mod.constructor()
assert mine.a == 2
c = mytype_mod.plus_b(mine, b=3)
assert c == 5

from othertype import othertype_mod

other = othertype_mod.constructor()
assert other.a == 5
d = othertype_mod.plus_b(other, b=4)
assert d == 9
