import test

test.wrap(1) # default only
test.wrap(1, 2) # default and optional
test.wrap(def_=1, opt=3) # kwarg form