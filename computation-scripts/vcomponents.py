# !! THIS FILE IS AUTOMATICALLY GENERATED.
# DO NOT EDIT.
# SEE calculation-helpers/generate.py

from typing import List, Tuple
import numpy as np


def v(
    U: float,
    t: float,
    epsl: float,
    occ_l_up: int,
    occ_l_down: int,
    neighbors_eps_occupation_tuples: List[Tuple[float, int, int]],
) -> np.complex128:
    res: np.complex128 = np.complex128(0)
    for (
        epsm,
        occ_m_up,
        occ_m_down,
    ) in neighbors_eps_occupation_tuples:
        if occ_l_up:
            if occ_l_down:
                if occ_m_up:
                    if occ_m_down:
                        # Lc:True, Mc:True, Ld:True, Md:True
                        res += 0
                    else:
                        # Lc:True, Mc:True, Ld:True, Md:False
                        res += 0 + (np.expm1(1j * (epsl - epsm) * t) / (epsl - epsm))
                else:
                    if occ_m_down:
                        # Lc:True, Mc:False, Ld:True, Md:True
                        res += 0 + (np.expm1(1j * (epsl - epsm) * t) / (epsl - epsm))
                    else:
                        # Lc:True, Mc:False, Ld:True, Md:False
                        res += 0 + 2 * (
                            np.expm1(1j * (epsl - epsm + U) * t) / (epsl - epsm + U)
                        )
            else:
                if occ_m_up:
                    if occ_m_down:
                        # Lc:True, Mc:True, Ld:False, Md:True
                        res += 0
                    else:
                        # Lc:True, Mc:True, Ld:False, Md:False
                        res += 0
                else:
                    if occ_m_down:
                        # Lc:True, Mc:False, Ld:False, Md:True
                        res += 0 + (
                            np.expm1(1j * (epsl - epsm - U) * t) / (epsl - epsm - U)
                        )
                    else:
                        # Lc:True, Mc:False, Ld:False, Md:False
                        res += 0 + (np.expm1(1j * (epsl - epsm) * t) / (epsl - epsm))
        else:
            if occ_l_down:
                if occ_m_up:
                    if occ_m_down:
                        # Lc:False, Mc:True, Ld:True, Md:True
                        res += 0
                    else:
                        # Lc:False, Mc:True, Ld:True, Md:False
                        res += 0 + (
                            np.expm1(1j * (epsl - epsm - U) * t) / (epsl - epsm - U)
                        )
                else:
                    if occ_m_down:
                        # Lc:False, Mc:False, Ld:True, Md:True
                        res += 0
                    else:
                        # Lc:False, Mc:False, Ld:True, Md:False
                        res += 0 + (np.expm1(1j * (epsl - epsm) * t) / (epsl - epsm))
            else:
                if occ_m_up:
                    if occ_m_down:
                        # Lc:False, Mc:True, Ld:False, Md:True
                        res += 0
                    else:
                        # Lc:False, Mc:True, Ld:False, Md:False
                        res += 0
                else:
                    if occ_m_down:
                        # Lc:False, Mc:False, Ld:False, Md:True
                        res += 0
                    else:
                        # Lc:False, Mc:False, Ld:False, Md:False
                        res += 0
    return res
