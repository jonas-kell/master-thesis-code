import time
from analyticallogicalfunctions import (
    part_A_flipping as flipping_A_pure_logic,
    part_B_flipping as flipping_B_pure_logic,
    part_C_flipping as flipping_C_pure_logic,
    part_A_flipping_if as flipping_A_pure_logic_if,
    part_B_flipping_if as flipping_B_pure_logic_if,
    part_C_flipping_if as flipping_C_pure_logic_if,
)
from logicalcalcfunctions import (
    part_A_flipping as flipping_A_espresso,
    part_B_flipping as flipping_B_espresso,
    part_C_flipping as flipping_C_espresso,
)
from random import randrange


# ! validity check
start = time.time()
for i in range(200000):
    a = randrange(2)
    b = randrange(2)
    c = randrange(2)
    d = randrange(2)
    e = randrange(2) == 1
    f = randrange(2) == 1
    if flipping_A_pure_logic(a, b, c, d, e, f) != flipping_A_espresso(
        a, b, c, d, e, f
    ) or flipping_A_espresso(a, b, c, d, e, f) != flipping_A_pure_logic_if(
        a, b, c, d, e, f
    ):
        print("alarm: A")
        print(a, b, c, d, e, f)
        exit()
    if flipping_B_pure_logic(a, b, c, d, e, f) != flipping_B_espresso(
        a, b, c, d, e, f
    ) or flipping_B_espresso(a, b, c, d, e, f) != flipping_B_pure_logic_if(
        a, b, c, d, e, f
    ):
        print("alarm: B")
        print(a, b, c, d, e, f)
        exit()
    if flipping_C_pure_logic(a, b, c, d, e, f) != flipping_C_espresso(
        a, b, c, d, e, f
    ) or flipping_C_espresso(a, b, c, d, e, f) != flipping_C_pure_logic_if(
        a, b, c, d, e, f
    ):
        print("alarm: C")
        print(a, b, c, d, e, f)
        exit()
end = time.time()
print(f"took {end-start:.4}s")
