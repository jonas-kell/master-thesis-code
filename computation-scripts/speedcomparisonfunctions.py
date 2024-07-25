import time
from analyticallogicalfunctions import (
    part_A_flipping as flipping_A_pure_logic,
    part_B_flipping as flipping_B_pure_logic,
    part_C_flipping as flipping_C_pure_logic,
)
from logicalcalcfunctions import (
    part_A_flipping as flipping_A_espresso,
    part_B_flipping as flipping_B_espresso,
    part_C_flipping as flipping_C_espresso,
)
from random import randrange

start = time.time()

for i in range(1000):
    a = randrange(2)
    b = randrange(2)
    c = randrange(2)
    d = randrange(2)
    e = randrange(2) == 1
    f = randrange(2) == 1

    if flipping_A_pure_logic(a, b, c, d, e, f) != flipping_A_espresso(a, b, c, d, e, f):
        print("alarm")
        print(a, b, c, d, e, f)
        exit()

    # if flipping_B_pure_logic(a, b, c, d, e, f) != flipping_B_espresso(a, b, c, d, e, f):
    #     print("alarm")
    #     print(a,b,c,d,e,f)

    # if flipping_C_pure_logic(a, b, c, d, e, f) != flipping_C_espresso(a, b, c, d, e, f):
    #     print("alarm")
    #     print(a,b,c,d,e,f)

end = time.time()

print(f"took {end-start:.4}s")
