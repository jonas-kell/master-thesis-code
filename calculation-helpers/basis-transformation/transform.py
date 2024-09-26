from typing import Tuple, Literal, List
from sympy import symbols, conjugate, Matrix, nsimplify, simplify, pprint
from sympy.core.sympify import sympify

BASIS_ELEMENT = Tuple[bool, bool, Literal["uu", "dd", "ud", "du"]]
BASIS = Tuple[
    List[BASIS_ELEMENT],
    List[BASIS_ELEMENT],
    List[BASIS_ELEMENT],
    List[BASIS_ELEMENT],
]


def getBraFromKet(ketBasis: BASIS):
    def invertElem(elem: BASIS_ELEMENT) -> BASIS_ELEMENT:
        has_i, plus, vec = elem
        res_plus = plus if (not has_i) else (not plus)
        return (has_i, res_plus, vec)

    ua, ub, uc, ud = ketBasis

    at = [invertElem(x) for x in ua]
    bt = [invertElem(x) for x in ub]
    ct = [invertElem(x) for x in uc]
    dt = [invertElem(x) for x in ud]

    return (at, bt, ct, dt)


def printBasis(basis: BASIS):
    for elem in basis:
        for index, summand in enumerate(elem):
            has_i, plus, vec = summand
            print(f"   {' ' if plus else '-'}{'i' if has_i else ' '}|{vec}|", end="")
            if index < len(elem) - 1:
                print("   +", end="")
        print()
    print()


def sFactor(j: int, i: int, basisFromKet, basisToKet):
    basisToBra = getBraFromKet(basisToKet)

    left = basisToBra[i]
    right = basisFromKet[j]

    out = 0.0
    for leftElem in left:
        has_i_left, plus_left, vec_left = leftElem
        for rightElem in right:
            has_i_right, plus_right, vec_right = rightElem

            if vec_left == vec_right:
                leftFactor = (1 if plus_left else -1) * (1j if has_i_left else 1)
                rightFactor = (1 if plus_right else -1) * (1j if has_i_right else 1)

                out += leftFactor * rightFactor

    return out


def format_complex(c):
    real_part = c.real
    imag_part = c.imag

    # Handle real and imaginary parts
    real_str = "" if real_part == 0 else f"{real_part:.0f}"

    if imag_part == 0:
        imag_str = ""
    elif imag_part == 1:
        imag_str = "i"
    elif imag_part == -1:
        imag_str = "-i"
    else:
        imag_str = f"{imag_part:.0f}i"

    # Combine parts properly
    if real_str and imag_str:
        sign = "+" if imag_part > 0 else ""
        return f"{real_str}{sign}{imag_str}"
    elif real_str:
        return real_str
    elif imag_str:
        return imag_str
    else:
        return "0"


def transformOperatorElement(
    j: int, l: int, op: Matrix, basisFrom: BASIS, basisTo: BASIS
):
    out = sympify(0)

    for i in range(4):
        for k in range(4):
            sji = sFactor(j, i, basisFrom, basisTo)
            slk_star = sFactor(l, k, basisFrom, basisTo).conjugate()

            factor = sji * slk_star

            if factor != 0:
                out += sympify(factor) * op[i * op.shape[0] + k]

    return out


def transformOperator(op: Matrix, basisFrom: BASIS, basisTo: BASIS):
    return (
        Matrix(
            [
                [
                    transformOperatorElement(j_row, i_col, op, basisFrom, basisTo)
                    for i_col in range(4)
                ]
                for j_row in range(4)
            ]
        )
        .applyfunc(nsimplify)
        .applyfunc(simplify)
    )


if __name__ == "__main__":
    standardBasis: BASIS = [
        [(False, True, "uu")],
        [(False, True, "ud")],
        [(False, True, "du")],
        [(False, True, "dd")],
    ]
    magicBasis: BASIS = [
        [(False, True, "uu"), (False, True, "dd")],
        [(True, True, "uu"), (True, False, "dd")],
        [(True, True, "ud"), (True, True, "du")],
        [(False, True, "ud"), (False, False, "du")],
    ]

    print("Standard Basis")
    printBasis(standardBasis)

    print("Magic Basis")
    printBasis(magicBasis)

    print("S_ji")
    for j_row in range(4):
        for i_col in range(4):
            print(f"{sFactor(j_row, i_col, standardBasis, magicBasis)}  ", end="")
        print("")
    print()

    print("q_ji")
    operator = Matrix(
        [
            [symbols("a", real=True), symbols("b"), symbols("c"), symbols("d")],
            [
                conjugate(symbols("b")),
                symbols("e", real=True),
                symbols("f"),
                symbols("g"),
            ],
            [
                conjugate(symbols("c")),
                conjugate(symbols("f")),
                symbols("h", real=True),
                symbols("k"),
            ],
            [
                conjugate(symbols("d")),
                conjugate(symbols("g")),
                conjugate(symbols("k")),
                symbols("l", real=True),
            ],
        ]
    )
    print(operator)
    print()

    print("q_ji in transformed basis")
    transformedOperator = transformOperator(operator, standardBasis, magicBasis)
    print(transformedOperator)
    print()

    print("complex conjugation in transformed basis")
    conjugation = conjugate(transformedOperator)
    print(conjugation)
    print()

    print("conjugation transformed back")
    backTransformedOperator = (
        (transformOperator(conjugation, magicBasis, standardBasis) / 4)
        .applyfunc(nsimplify)
        .applyfunc(simplify)
    )
    print(backTransformedOperator)
    print()

    print("Spin flip operation result")
    pprint(operator)
    print()
    pprint(backTransformedOperator)
