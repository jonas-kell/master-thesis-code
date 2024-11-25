from typing import List
from common import indent, generate_if_tree, write_file, init_file, is_basically_one


def checkOptimizations(inputMappings):
    for key, mappings in inputMappings.items():
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


def generateHelperFile(inputMappings):
    init_file(FILENAME)

    # normal V
    write_file(
        FILENAME,
        "def v(U: float, t: float, epsl: float, occ_l_up: int, occ_l_down: int,neighbors_eps_occupation_tuples: List[Tuple[float, int, int]],) -> np.complex128:\n",
    )
    write_file(FILENAME, indent(1) + "res: np.complex128 = np.complex128(0)\n")
    write_file(
        FILENAME,
        indent(1)
        + "for (epsm, occ_m_up, occ_m_down,) in neighbors_eps_occupation_tuples:\n",
    )

    def endCallback(lineStart: str, currentTruthinesses: List[bool]):
        Lc, Ld, Mc, Md = currentTruthinesses

        res = ""
        res += lineStart + f"# Lc:{Lc}, Mc:{Mc}, Ld:{Ld}, Md:{Md}" + "\n"
        res += lineStart + "res += 0 "
        for meta, mappings in inputMappings.values():
            mult = mappings[0](Lc, Mc, Ld, Md)
            if mult != 0:
                if is_basically_one(mult):
                    res += "+" + meta[0]
                elif is_basically_one(-mult):
                    res += "-" + meta[0]
                else:
                    res += "+ " + str(mult) + " * " + meta[0]
        res += "\n"
        return res

    write_file(
        FILENAME,
        generate_if_tree(
            2, ["occ_l_up", "occ_l_down", "occ_m_up", "occ_m_down"], endCallback
        ),
    )

    write_file(FILENAME, indent(1) + "return res\n\n\n")

    # V, flipping
    write_file(
        FILENAME,
        "def v_flip(flip_up: bool, U: float, t: float, epsl: float, occ_l_up: int, occ_l_down: int,neighbors_eps_occupation_tuples: List[Tuple[float, int, int]],) -> np.complex128:\n",
    )
    write_file(FILENAME, indent(1) + "res: np.complex128 = np.complex128(0)\n")
    write_file(
        FILENAME,
        indent(1)
        + "for (epsm, occ_m_up, occ_m_down,) in neighbors_eps_occupation_tuples:\n",
    )

    def endCallbackFlip(lineStart: str, currentTruthinesses: List[bool]):
        flipUp, Lc, Ld, Mc, Md = currentTruthinesses

        res = ""
        res += (
            lineStart + f"# flipUp:{flipUp}, Lc:{Lc}, Mc:{Mc}, Ld:{Ld}, Md:{Md}" + "\n"
        )
        res += lineStart + "res += 0 "
        for meta, mappings in inputMappings.values():
            LcPrime = (1 - Lc) if flipUp else Lc
            McPrime = Mc
            LdPrime = (1 - Ld) if not flipUp else Ld
            MdPrime = Md

            # first meta entry - normal
            mult = mappings[0](Lc, Mc, Ld, Md) - mappings[0](
                LcPrime, McPrime, LdPrime, MdPrime
            )
            if mult != 0:
                if is_basically_one(mult):
                    res += "+" + meta[0]
                elif is_basically_one(-mult):
                    res += "-" + meta[0]
                else:
                    res += "+ " + str(mult) + " * " + meta[0]
            # second meta entry - l<->m swapped
            mult = mappings[0](Mc, Lc, Md, Ld) - mappings[0](
                McPrime, LcPrime, MdPrime, LdPrime
            )
            if mult != 0:
                if is_basically_one(mult):
                    res += "+" + meta[1]
                elif is_basically_one(-mult):
                    res += "-" + meta[1]
                else:
                    res += "+ " + str(mult) + " * " + meta[1]

        res += "\n"
        return res

    write_file(
        FILENAME,
        generate_if_tree(
            2,
            ["flip_up", "occ_l_up", "occ_l_down", "occ_m_up", "occ_m_down"],
            endCallbackFlip,
        ),
    )

    write_file(FILENAME, indent(1) + "return res\n\n\n")

    # V, hopping
    write_file(
        FILENAME,
        "def v_hop(hop_sw1_up: bool, hop_sw2_up: bool, U: float, t: float, eps_sw1: float, occ_sw1_up: int, occ_sw1_down: int, occ_sw2_up: int, occ_sw2_down: int, neighbors_eps_occupation_tuples: List[Tuple[float, int, int, bool]],) -> np.complex128:\n",
    )
    write_file(FILENAME, indent(1) + "res: np.complex128 = np.complex128(0)\n")
    write_file(
        FILENAME,
        indent(1)
        + "for (eps_nb, occ_nb_up, occ_nb_down, direct_swap) in neighbors_eps_occupation_tuples:\n",
    )

    def endCallbackHop(lineStart: str, currentTruthinesses: List[bool]):
        DirectSwap, HopLUp, HopMUp, sw1C, sw1D, sw2C, sw2D, nbC, nbD = (
            currentTruthinesses
        )

        res = ""
        res += (
            lineStart
            + f"# DirectSwap:{DirectSwap}, HopLUp:{HopLUp}, HopMUp:{HopMUp}, sw1C:{sw1C}, sw1D:{sw1D}, sw2C:{sw2C}, sw2D:{sw2D}, nbC:{nbC}, nbD:{nbD}"
            + "\n"
        )
        res += lineStart + "res += 0 "
        for meta, mappings in inputMappings.values():
            LcPrime = (sw2C if HopMUp else sw2D) if HopLUp else sw1C
            LdPrime = (sw2D if not HopMUp else sw2C) if not HopLUp else sw1D
            if DirectSwap:
                McPrime = (sw1C if HopLUp else sw1D) if HopMUp else sw2C
                MdPrime = (sw1D if not HopLUp else sw1C) if not HopMUp else sw2D
            else:
                McPrime = nbC
                MdPrime = nbD

            # first meta entry - normal
            mult = mappings[0](sw1C, nbC, sw1D, nbD) - mappings[0](
                LcPrime, McPrime, LdPrime, MdPrime
            )
            if mult != 0:
                if is_basically_one(mult):
                    res += "+" + meta[0]
                elif is_basically_one(-mult):
                    res += "-" + meta[0]
                else:
                    res += "+ " + str(mult) + " * " + meta[0]
            if not DirectSwap:
                # would otherwise doubly apply this one as one is reverse of other and vice versa
                # second meta entry - l<->m swapped
                mult = mappings[0](nbC, sw1C, nbD, sw1D) - mappings[0](
                    McPrime, LcPrime, MdPrime, LdPrime
                )
                if mult != 0:
                    if is_basically_one(mult):
                        res += "+" + meta[1]
                    elif is_basically_one(-mult):
                        res += "-" + meta[1]
                    else:
                        res += "+ " + str(mult) + " * " + meta[1]

        res = res.replace("epsl", "eps_sw1")
        res = res.replace("epsm", "eps_nb")

        res += "\n"
        return res

    write_file(
        FILENAME,
        generate_if_tree(
            2,
            [
                "direct_swap",
                "hop_sw1_up",
                "hop_sw2_up",
                "occ_sw1_up",
                "occ_sw1_down",
                "occ_sw2_up",
                "occ_sw2_down",
                "occ_nb_up",
                "occ_nb_down",
            ],
            endCallbackHop,
        ),
    )

    write_file(FILENAME, indent(1) + "return res\n\n\n")

    # V, double flipping
    write_file(
        FILENAME,
        "def v_double_flip(flip1_up: bool, flip2_up: bool, U: float, t: float, flip1_eps: float, flip1_occ_up: int, flip1_occ_down: int, neighbors_eps_occupation_tuples: List[Tuple[float, int, int, bool]]) -> np.complex128:\n",
    )
    write_file(FILENAME, indent(1) + "res: np.complex128 = np.complex128(0)\n")
    write_file(
        FILENAME,
        indent(1)
        + "for (nb_eps, nb_occ_up, nb_occ_down,direct) in neighbors_eps_occupation_tuples:\n",
    )

    def endCallbackDoubleFlip(lineStart: str, currentTruthinesses: List[bool]):
        Direct, flip1Up, flip2Up, Lc, Ld, Mc, Md = currentTruthinesses

        res = ""
        res += (
            lineStart
            + f"# Direct:{Direct}, flip1Up:{flip1Up}, flip2Up:{flip2Up}, Lc:{Lc}, Mc:{Mc}, Ld:{Ld}, Md:{Md}"
            + "\n"
        )
        res += lineStart + "res += 0 "
        for meta, mappings in inputMappings.values():
            LcPrime = (1 - Lc) if flip1Up else Lc
            LdPrime = (1 - Ld) if not flip1Up else Ld
            if Direct:
                McPrime = (1 - Mc) if flip2Up else Mc
                MdPrime = (1 - Md) if not flip2Up else Md
            else:
                McPrime = Mc
                MdPrime = Md

            # first meta entry - normal
            mult = mappings[0](Lc, Mc, Ld, Md) - mappings[0](
                LcPrime, McPrime, LdPrime, MdPrime
            )
            if mult != 0:
                if is_basically_one(mult):
                    res += "+" + meta[0]
                elif is_basically_one(-mult):
                    res += "-" + meta[0]
                else:
                    res += "+ " + str(mult) + " * " + meta[0]
            if not Direct:
                # would otherwise doubly apply this one as one is reverse of other and vice versa
                # second meta entry - l<->m swapped
                mult = mappings[0](Mc, Lc, Md, Ld) - mappings[0](
                    McPrime, LcPrime, MdPrime, LdPrime
                )
                if mult != 0:
                    if is_basically_one(mult):
                        res += "+" + meta[1]
                    elif is_basically_one(-mult):
                        res += "-" + meta[1]
                    else:
                        res += "+ " + str(mult) + " * " + meta[1]

        res = res.replace("epsl", "flip1_eps")
        res = res.replace("epsm", "nb_eps")

        res += "\n"
        return res

    write_file(
        FILENAME,
        generate_if_tree(
            2,
            [
                "direct",
                "flip1_up",
                "flip2_up",
                "flip1_occ_up",
                "flip1_occ_down",
                "nb_occ_up",
                "nb_occ_down",
            ],
            endCallbackDoubleFlip,
        ),
    )

    write_file(FILENAME, indent(1) + "return res\n\n\n")

    # V, double flipping, same site
    write_file(
        FILENAME,
        "def v_double_flip_same_site(U: float, t: float, flip_eps: float, flip_occ_up: int, flip_occ_down: int, neighbors_eps_occupation_tuples: List[Tuple[float, int, int]]) -> np.complex128:\n",
    )
    write_file(FILENAME, indent(1) + "res: np.complex128 = np.complex128(0)\n")
    write_file(
        FILENAME,
        indent(1)
        + "for (nb_eps, nb_occ_up, nb_occ_down) in neighbors_eps_occupation_tuples:\n",
    )

    def endCallbackDoubleSameSiteFlip(lineStart: str, currentTruthinesses: List[bool]):
        Lc, Ld, Mc, Md = currentTruthinesses

        res = ""
        res += lineStart + f"# Lc:{Lc}, Mc:{Mc}, Ld:{Ld}, Md:{Md}" + "\n"
        res += lineStart + "res += 0 "
        for meta, mappings in inputMappings.values():
            LcPrime = 1 - Lc
            LdPrime = 1 - Ld
            McPrime = Mc
            MdPrime = Md

            # first meta entry - normal
            mult = mappings[0](Lc, Mc, Ld, Md) - mappings[0](
                LcPrime, McPrime, LdPrime, MdPrime
            )
            if mult != 0:
                if is_basically_one(mult):
                    res += "+" + meta[0]
                elif is_basically_one(-mult):
                    res += "-" + meta[0]
                else:
                    res += "+ " + str(mult) + " * " + meta[0]
            # second meta entry - l<->m swapped
            mult = mappings[0](Mc, Lc, Md, Ld) - mappings[0](
                McPrime, LcPrime, MdPrime, LdPrime
            )
            if mult != 0:
                if is_basically_one(mult):
                    res += "+" + meta[1]
                elif is_basically_one(-mult):
                    res += "-" + meta[1]
                else:
                    res += "+ " + str(mult) + " * " + meta[1]

        res = res.replace("epsl", "flip_eps")
        res = res.replace("epsm", "nb_eps")

        res += "\n"
        return res

    write_file(
        FILENAME,
        generate_if_tree(
            2,
            [
                "flip_occ_up",
                "flip_occ_down",
                "nb_occ_up",
                "nb_occ_down",
            ],
            endCallbackDoubleSameSiteFlip,
        ),
    )

    write_file(FILENAME, indent(1) + "return res\n\n\n")


if __name__ == "__main__":
    mappingsDict = {
        "A": (
            (
                "(np.expm1(1j * (epsl - epsm) * t) / (epsl-epsm))",
                "(np.expm1(1j * (epsm - epsl) * t) / (epsm-epsl))",
            ),
            [
                lambda Lc, Mc, Ld, Md: Lc * (1 - Mc) * (1 + 2 * Ld * Md - Md - Ld)
                + Ld * (1 - Md) * (1 + 2 * Lc * Mc - Mc - Lc),
                lambda Lc, Mc, Ld, Md: (Lc and (1 - Mc) and Ld == Md)
                + (Ld and (1 - Md) and Lc == Mc),
            ],
        ),
        "B": (
            (
                "(np.expm1(1j * (epsl - epsm+U) * t) / (epsl-epsm+U))",
                "(np.expm1(1j * (epsm - epsl+U) * t) / (epsm-epsl+U))",
            ),
            [
                lambda Lc, Mc, Ld, Md: Lc * (1 - Mc) * Ld * (1 - Md)
                + Ld * (1 - Md) * Lc * (1 - Mc)
            ],
        ),
        "C": (
            (
                "(np.expm1(1j * (epsl - epsm-U) * t) / (epsl-epsm-U))",
                "(np.expm1(1j * (epsm - epsl-U) * t) / (epsm-epsl-U))",
            ),
            [
                lambda Lc, Mc, Ld, Md: Lc * (1 - Mc) * Md * (1 - Ld)
                + Ld * (1 - Md) * Mc * (1 - Lc)
            ],
        ),
    }

    checkSymmetry(mappingsDict)
    checkOptimizations(mappingsDict)
    generateHelperFile(mappingsDict)
