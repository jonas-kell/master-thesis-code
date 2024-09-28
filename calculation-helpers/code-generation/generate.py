from typing import List


def checkOptimizations(inputMappings):
    for key, mappings in inputMappings.items():
        print("\n\n\n" + key)

        compLambda = mappings[1][0]
        for mappingLambda in mappings[1]:
            for Lc in range(2):
                for Mc in range(2):
                    for Ld in range(2):
                        for Md in range(2):
                            if compLambda(Lc, Mc, Ld, Md) != mappingLambda(
                                Lc, Mc, Ld, Md
                            ):
                                raise Exception(f"Difference found at {key}")


def checkSymmetry(inputMappings):
    for Lc in range(2):
        for Mc in range(2):
            for Ld in range(2):
                for Md in range(2):
                    for key, mappings in inputMappings.items():
                        mainMapping = mappings[1][0]
                        res = mainMapping(Lc, Mc, Ld, Md)
                        res2 = mainMapping(Ld, Md, Lc, Mc)

                        if res != res2:
                            raise Exception(f"{key} not symmetric at {Lc}{Mc}{Ld}{Md}")


FILENAME = "./../../computation-scripts/vcomponents.py"


# This function marks the GENERATED FILE as not to be edited manually
def init_file():
    text = """# !! THIS FILE IS AUTOMATICALLY GENERATED.
# DO NOT EDIT. 
# SEE calculation-helpers/generate.py

from typing import List, Tuple
import numpy as np

"""
    with open(FILENAME, "w") as file:
        file.write(text)


def write_file(input: str):
    with open(FILENAME, "a") as file:
        file.write(input)


def indent(spaces: int):
    return "    " * spaces


def generateIfTree(
    initialIndent: int,
    checks: List[str],
    callbackForTop=lambda a, b: a + "pass\n",
    currentDepth: int = 0,
    currentTruthinesses: List[bool] = [],
) -> str:
    maxDepth = len(checks)

    if currentDepth == maxDepth:
        return callbackForTop(indent(initialIndent + currentDepth), currentTruthinesses)
    else:
        res = indent(initialIndent + currentDepth) + f"if {checks[currentDepth]}:\n"
        res += generateIfTree(
            initialIndent,
            checks,
            callbackForTop,
            currentDepth + 1,
            currentTruthinesses + [True],
        )
        res += indent(initialIndent + currentDepth) + "else:\n"
        res += generateIfTree(
            initialIndent,
            checks,
            callbackForTop,
            currentDepth + 1,
            currentTruthinesses + [False],
        )
        return res


def generateHelperFile(inputMappings):
    init_file()

    # normal V
    write_file(
        "def v(U: float, t: float, epsl: float, occ_l_up: int, occ_l_down: int,neighbors_eps_occupation_tuples: List[Tuple[float, int, int]],) -> np.complex128:\n"
    )
    write_file(indent(1) + "res: np.complex128 = np.complex128(0)\n")
    write_file(
        indent(1)
        + "for (epsm, occ_m_up, occ_m_down,) in neighbors_eps_occupation_tuples:\n"
    )

    def endCallback(lineStart: str, currentTruthinesses: List[bool]):
        Lc, Ld, Mc, Md = currentTruthinesses

        res = ""
        res += lineStart + f"# Lc:{Lc}, Mc:{Mc}, Ld:{Ld}, Md:{Md}" + "\n"
        res += lineStart + "res += 0 "
        for meta, mappings in inputMappings.values():
            mult = mappings[0](Lc, Mc, Ld, Md)
            if mult != 0:
                if mult == 1:
                    res += "+" + meta
                else:
                    res += "+ " + str(mult) + " * " + meta
        res += "\n"
        return res

    write_file(
        generateIfTree(
            2, ["occ_l_up", "occ_l_down", "occ_m_up", "occ_m_down"], endCallback
        )
    )

    write_file(indent(1) + "return res\n\n\n")


mappingsDict = {
    "A": (
        "(np.expm1(1j * (epsl - epsm) * t) / (epsm-epsl))",
        [
            lambda Lc, Mc, Ld, Md: Lc * (1 - Mc) * (1 + 2 * Ld * Md - Md - Ld)
            + Ld * (1 - Md) * (1 + 2 * Lc * Mc - Mc - Lc),
            lambda Lc, Mc, Ld, Md: (Lc and (1 - Mc) and Ld == Md)
            + (Ld and (1 - Md) and Lc == Mc),
        ],
    ),
    "B": (
        "(np.expm1(1j * (epsl - epsm+U) * t) / (epsm-epsl+U))",
        [
            lambda Lc, Mc, Ld, Md: Lc * (1 - Mc) * Ld * (1 - Md)
            + Ld * (1 - Md) * Lc * (1 - Mc)
        ],
    ),
    "C": (
        "(np.expm1(1j * (epsl - epsm-U) * t) / (epsm-epsl-U))",
        [
            lambda Lc, Mc, Ld, Md: Lc * (1 - Mc) * Md * (1 - Ld)
            + Ld * (1 - Md) * Mc * (1 - Lc)
        ],
    ),
}


if __name__ == "__main__":
    checkSymmetry(mappingsDict)
    checkOptimizations(mappingsDict)
    generateHelperFile(mappingsDict)
