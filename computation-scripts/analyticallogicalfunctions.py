# python3 treats (1 == True) -> True and (0 == False) -> True


def part_A_flipping_if_free(
    Lu: int, Ld: int, Mu: int, Md: int, flip_up: bool, flip_l: bool
) -> int:

    # ! make sure to clone block to all the same functions, inline for speed
    # controlled negation of one of the one var decided by flip_up/flip_l, purely computational, no jumps (probably slower in python)
    Lup = Lu == (not (flip_up and flip_l))
    Mup = Mu == (not flip_up or flip_l)  # (not (flip_up and not flip_l))
    Ldp = Ld == (flip_up or not flip_l)  # (not (not flip_up and flip_l))
    Mdp = Md == (flip_up or flip_l)  # (not (not flip_up and not flip_l))
    # ! make sure to clone block to all the same functions, inline for speed

    res = 0
    res += Lu and not Mu and Ld == Md
    res -= Lup and not Mup and Ldp == Mdp
    res += Ld and not Md and Lu == Mu
    res -= Ldp and not Mdp and Lup == Mup

    return res


def part_B_flipping_if_free(
    Lu: int, Ld: int, Mu: int, Md: int, flip_up: bool, flip_l: bool
) -> int:

    # ! make sure to clone block to all the same functions, inline for speed
    # controlled negation of one of the one var decided by flip_up/flip_l, purely computational, no jumps (probably slower in python)
    Lup = Lu == (not (flip_up and flip_l))
    Mup = Mu == (not flip_up or flip_l)  # (not (flip_up and not flip_l))
    Ldp = Ld == (flip_up or not flip_l)  # (not (not flip_up and flip_l))
    Mdp = Md == (flip_up or flip_l)  # (not (not flip_up and not flip_l))
    # ! make sure to clone block to all the same functions, inline for speed

    res = 0
    res += Lu and not Mu and not Md and Ld
    res -= Lup and not Mup and not Mdp and Ldp
    res += Ld and not Md and not Mu and Lu
    res -= Ldp and not Mdp and not Mup and Lup

    return res


def part_C_flipping_if_free(
    Lu: int, Ld: int, Mu: int, Md: int, flip_up: bool, flip_l: bool
) -> int:

    # ! make sure to clone block to all the same functions, inline for speed
    # controlled negation of one of the one var decided by flip_up/flip_l, purely computational, no jumps (probably slower in python)
    Lup = Lu == (not (flip_up and flip_l))
    Mup = Mu == (not flip_up or flip_l)  # (not (flip_up and not flip_l))
    Ldp = Ld == (flip_up or not flip_l)  # (not (not flip_up and flip_l))
    Mdp = Md == (flip_up or flip_l)  # (not (not flip_up and not flip_l))
    # ! make sure to clone block to all the same functions, inline for speed

    res = 0
    res += Lu and not Mu and Md and not Ld
    res -= Lup and not Mup and Mdp and not Ldp
    res += Ld and not Md and Mu and not Lu
    res -= Ldp and not Mdp and Mup and not Lup

    return res


def part_A_flipping(
    Lu: int, Ld: int, Mu: int, Md: int, flip_up: bool, flip_l: bool
) -> int:

    # ! make sure to clone block to all the same functions, inline for speed
    if flip_up:
        if flip_l:
            Lup = not Lu
            Ldp = Ld
            Mup = Mu
            Mdp = Md
        else:
            Lup = Lu
            Ldp = Ld
            Mup = not Mu
            Mdp = Md
    else:
        if flip_l:
            Lup = Lu
            Ldp = not Ld
            Mup = Mu
            Mdp = Md
        else:
            Lup = Lu
            Ldp = Ld
            Mup = Mu
            Mdp = not Md
    # ! make sure to clone block to all the same functions, inline for speed

    res = 0
    res += Lu and not Mu and Ld == Md
    res -= Lup and not Mup and Ldp == Mdp
    res += Ld and not Md and Lu == Mu
    res -= Ldp and not Mdp and Lup == Mup

    return res


def part_B_flipping(
    Lu: int, Ld: int, Mu: int, Md: int, flip_up: bool, flip_l: bool
) -> int:

    # ! make sure to clone block to all the same functions, inline for speed
    if flip_up:
        if flip_l:
            Lup = not Lu
            Ldp = Ld
            Mup = Mu
            Mdp = Md
        else:
            Lup = Lu
            Ldp = Ld
            Mup = not Mu
            Mdp = Md
    else:
        if flip_l:
            Lup = Lu
            Ldp = not Ld
            Mup = Mu
            Mdp = Md
        else:
            Lup = Lu
            Ldp = Ld
            Mup = Mu
            Mdp = not Md
    # ! make sure to clone block to all the same functions, inline for speed

    res = 0
    res += Lu and not Mu and not Md and Ld
    res -= Lup and not Mup and not Mdp and Ldp
    res += Ld and not Md and not Mu and Lu
    res -= Ldp and not Mdp and not Mup and Lup

    return res


def part_C_flipping(
    Lu: int, Ld: int, Mu: int, Md: int, flip_up: bool, flip_l: bool
) -> int:

    # ! make sure to clone block to all the same functions, inline for speed
    if flip_up:
        if flip_l:
            Lup = not Lu
            Ldp = Ld
            Mup = Mu
            Mdp = Md
        else:
            Lup = Lu
            Ldp = Ld
            Mup = not Mu
            Mdp = Md
    else:
        if flip_l:
            Lup = Lu
            Ldp = not Ld
            Mup = Mu
            Mdp = Md
        else:
            Lup = Lu
            Ldp = Ld
            Mup = Mu
            Mdp = not Md
    # ! make sure to clone block to all the same functions, inline for speed

    res = 0
    res += Lu and not Mu and Md and not Ld
    res -= Lup and not Mup and Mdp and not Ldp
    res += Ld and not Md and Mu and not Lu
    res -= Ldp and not Mdp and Mup and not Lup

    return res


def part_A_hopping(
    Lu: int,
    Ld: int,
    Mu: int,
    Md: int,
    I: int,
    swp_i_l: bool | None,
    i_up: bool,
    J: int,
    j_up: bool,
    swp_j_l: bool | None,
) -> int:

    # ! make sure to clone block to all the same functions, inline for speed
    if swp_i_l is not None:
        if swp_i_l:
            if i_up:
                Lup = I
                Ldp = Ld
                Mup = Mu
                Mdp = Md
            else:
                Lup = Lu
                Ldp = I
                Mup = Mu
                Mdp = Md
        else:
            if i_up:
                Lup = Lu
                Ldp = Ld
                Mup = I
                Mdp = Md
            else:
                Lup = Lu
                Ldp = Ld
                Mup = Mu
                Mdp = I
    else:
        Lup = Lu
        Ldp = Ld
        Mup = Mu
        Mdp = I
    if swp_j_l is not None:
        if swp_j_l:
            if j_up:
                Lup = J
            else:
                Ldp = J
        else:
            if j_up:
                Mup = J
            else:
                Mdp = J
    # ! make sure to clone block to all the same functions, inline for speed

    res = 0
    res += Lu and not Mu and Ld == Md
    res -= Lup and not Mup and Ldp == Mdp
    res += Ld and not Md and Lu == Mu
    res -= Ldp and not Mdp and Lup == Mup

    return res


def part_B_hopping(
    Lu: int,
    Ld: int,
    Mu: int,
    Md: int,
    I: int,
    swp_i_l: bool | None,
    i_up: bool,
    J: int,
    j_up: bool,
    swp_j_l: bool | None,
) -> int:

    # ! make sure to clone block to all the same functions, inline for speed
    if swp_i_l is not None:
        if swp_i_l:
            if i_up:
                Lup = I
                Ldp = Ld
                Mup = Mu
                Mdp = Md
            else:
                Lup = Lu
                Ldp = I
                Mup = Mu
                Mdp = Md
        else:
            if i_up:
                Lup = Lu
                Ldp = Ld
                Mup = I
                Mdp = Md
            else:
                Lup = Lu
                Ldp = Ld
                Mup = Mu
                Mdp = I
    else:
        Lup = Lu
        Ldp = Ld
        Mup = Mu
        Mdp = I
    if swp_j_l is not None:
        if swp_j_l:
            if j_up:
                Lup = J
            else:
                Ldp = J
        else:
            if j_up:
                Mup = J
            else:
                Mdp = J
    # ! make sure to clone block to all the same functions, inline for speed

    res = 0
    res += Lu and not Mu and not Md and Ld
    res -= Lup and not Mup and not Mdp and Ldp
    res += Ld and not Md and not Mu and Lu
    res -= Ldp and not Mdp and not Mup and Lup

    return res


def part_C_hopping(
    Lu: int,
    Ld: int,
    Mu: int,
    Md: int,
    I: int,
    swp_i_l: bool | None,
    i_up: bool,
    J: int,
    j_up: bool,
    swp_j_l: bool | None,
) -> int:

    # ! make sure to clone block to all the same functions, inline for speed
    if swp_i_l is not None:
        if swp_i_l:
            if i_up:
                Lup = I
                Ldp = Ld
                Mup = Mu
                Mdp = Md
            else:
                Lup = Lu
                Ldp = I
                Mup = Mu
                Mdp = Md
        else:
            if i_up:
                Lup = Lu
                Ldp = Ld
                Mup = I
                Mdp = Md
            else:
                Lup = Lu
                Ldp = Ld
                Mup = Mu
                Mdp = I
    else:
        Lup = Lu
        Ldp = Ld
        Mup = Mu
        Mdp = I
    if swp_j_l is not None:
        if swp_j_l:
            if j_up:
                Lup = J
            else:
                Ldp = J
        else:
            if j_up:
                Mup = J
            else:
                Mdp = J
    # ! make sure to clone block to all the same functions, inline for speed

    res = 0
    res += Lu and not Mu and Md and not Ld
    res -= Lup and not Mup and Mdp and not Ldp
    res += Ld and not Md and Mu and not Lu
    res -= Ldp and not Mdp and Mup and not Lup

    return res
