from typing import Tuple, Literal, List
import numpy as np

BRA_KET_THINGY = Literal["u", "d"]
# plus/minus, has_i/has_not_i, vec
MATRIX_ELEMENT_SUBVECTOR_SPACE = Tuple[bool, bool, BRA_KET_THINGY, BRA_KET_THINGY]
MATRIX_SUBVECTOR_SPACE = List[MATRIX_ELEMENT_SUBVECTOR_SPACE]


def printMatrixSubvector(sv: MATRIX_SUBVECTOR_SPACE):
    for index, summand in enumerate(sv):
        plus, has_i, udket, udbra = summand
        print(
            f"   {' ' if plus else '-'}{'i' if has_i else ' '}|{udket}><{udbra}|",
            end="",
        )
        if index < len(sv) - 1:
            print("   +", end="")
    print()


def getMatrixInZBasis(
    sva: MATRIX_SUBVECTOR_SPACE, svb: MATRIX_SUBVECTOR_SPACE
) -> np.ndarray:
    indexTranslationMap = {"uu": 0, "ud": 1, "du": 2, "dd": 3}

    res = np.zeros((4, 4), dtype=np.complex64)

    for summandOfA in sva:
        plusA, has_iA, udketA, udbraA = summandOfA
        for summandOfB in svb:
            plusB, has_iB, udketB, udbraB = summandOfB

            extraMinus = has_iA and has_iB
            minus = ((not plusA) + (not plusB) + extraMinus) % 2 == 1
            has_i = (has_iA and not has_iB) or (not has_iA and has_iB)

            factor = (-1 if minus else 1) * (1j if has_i else 1)
            ket_index_row = indexTranslationMap[udketA + udketB]
            bra_index_col = indexTranslationMap[udbraA + udbraB]

            res[ket_index_row][bra_index_col] += factor

    return res


if __name__ == "__main__":
    o_spin: MATRIX_SUBVECTOR_SPACE = [
        (True, False, "u", "u"),
        (True, False, "d", "d"),
    ]
    printMatrixSubvector(o_spin)
    x_spin: MATRIX_SUBVECTOR_SPACE = [
        (True, False, "u", "d"),
        (True, False, "d", "u"),
    ]
    printMatrixSubvector(x_spin)
    y_spin: MATRIX_SUBVECTOR_SPACE = [
        (True, True, "d", "u"),
        (False, True, "u", "d"),
    ]
    printMatrixSubvector(y_spin)
    z_spin: MATRIX_SUBVECTOR_SPACE = [
        (True, False, "u", "u"),
        (False, False, "d", "d"),
    ]
    printMatrixSubvector(z_spin)

    print()
    print()
    print()

    print(getMatrixInZBasis(o_spin, o_spin))
    print(getMatrixInZBasis(o_spin, y_spin))
    print(getMatrixInZBasis(y_spin, y_spin))

    print()
    print()
    print()

    order = [o_spin, x_spin, y_spin, z_spin]

    # verify normalization factor
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    print(
                        i,
                        j,
                        k,
                        l,
                        np.trace(
                            getMatrixInZBasis(order[i], order[j])
                            @ getMatrixInZBasis(order[k], order[l])
                        ),
                    )

    print()
    print()
    print()

    # get the mapping array
    print("densityMatrixBuilderArray = np.array([")
    for i in range(4):
        print("[", end="")
        for j in range(4):
            matrix = getMatrixInZBasis(order[i], order[j])
            print("[", end="")
            for row in range(4):
                print("[", end="")
                for col in range(4):
                    print(f"{matrix[row, col]}", end="")
                    if col < 3:
                        print(",", end="")
                print("],", end="")
            print("],", end="")
        print("],")
    print("])")
