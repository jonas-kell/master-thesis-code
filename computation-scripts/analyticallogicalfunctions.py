def part_A_flipping(
    Lu: int, Ld: int, Mu: int, Md: int, flip_up: bool, flip_l: bool
) -> int:
    res = 0
    res += (
        (Lu and not Ld and not Mu and not Md)
        or (not Lu and Ld and not Mu and not Md)
        or (Lu and Ld and Mu and not Md)
        or (Lu and Ld and not Mu and Md)
    )
    res -= (
        (not Lu and not Ld and not Mu and not Md and not flip_l)
        or (not Lu and Ld and Mu and not Md and not flip_up)
        or (not Lu and Ld and not Mu and Md and not flip_up and not flip_l)
        or (Lu and not Ld and Mu and not Md and flip_up and not flip_l)
        or (Lu and not Ld and not Mu and Md and flip_up)
        or (Lu and not Ld and Mu and not Md and not flip_up and flip_l)
        or (Lu and Ld and Mu and Md and flip_l)
        or (not Lu and Ld and not Mu and Md and flip_up and flip_l)
        or (Lu and Ld and not Mu and not Md)
    )
    return res


def part_B_flipping(
    Lu: int, Ld: int, Mu: int, Md: int, flip_up: bool, flip_l: bool
) -> int:
    res = 0
    res += Lu and Ld and not Mu and not Md
    res -= (
        (not Lu and Ld and not Mu and not Md and not flip_up and not flip_l)
        or (Lu and not Ld and not Mu and not Md and flip_up and not flip_l)
        or (Lu and Ld and Mu and not Md and not flip_up and flip_l)
        or (Lu and Ld and not Mu and Md and flip_up and flip_l)
    )
    return 2 * res


def part_C_flipping(
    Lu: int, Ld: int, Mu: int, Md: int, flip_up: bool, flip_l: bool
) -> int:
    res = 0
    res += (not Lu and Ld and Mu and not Md) or (Lu and not Ld and not Mu and Md)
    res -= (
        (Lu and Ld and Mu and not Md and not flip_up and not flip_l)
        or (not Lu and not Ld and not Mu and Md and not flip_up and not flip_l)
        or (not Lu and not Ld and Mu and not Md and flip_up and not flip_l)
        or (Lu and Ld and not Mu and Md and flip_up and not flip_l)
        or (not Lu and Ld and not Mu and not Md and not flip_up and flip_l)
        or (Lu and not Ld and Mu and Md and not flip_up and flip_l)
        or (Lu and not Ld and not Mu and not Md and flip_up and flip_l)
        or (not Lu and Ld and Mu and Md and flip_up and flip_l)
    )
    return res
