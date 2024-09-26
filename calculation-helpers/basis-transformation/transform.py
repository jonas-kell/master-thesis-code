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
    basisFromBra = getBraFromKet(basisFromKet)

    left = basisFromBra[i]
    right = basisToKet[j]

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


def evalMatrix(expr, a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, k=0, l=0):
    pprint(
        expr.subs(
            {
                symbols("a", real=True): a,
                symbols("b"): b,
                symbols("c"): c,
                symbols("d"): d,
                symbols("e", real=True): e,
                symbols("f"): f,
                symbols("g"): g,
                symbols("h", real=True): h,
                symbols("k"): k,
                symbols("l", real=True): l,
            }
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

    # evalMatrix(operator, a=1)
    # print()
    # evalMatrix(transformedOperator, a=1)
