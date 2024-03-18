from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Union
from copy import deepcopy

UP = "↑"
DOWN = "↓"


class OccupationNumber(ABC):
    def __init__(
        self,
        index: str,
        spin: Union[Literal["↑"], Literal["↓"]],
    ):
        self.index = index
        self.spin = spin

    @abstractmethod
    def text_representation(self) -> str:
        pass

    def overwrite_index(self, index: str):
        self.index = index

    def overwrite_spin(self, spin: Union[Literal["↑"], Literal["↓"]]):
        self.spin = spin


class Occupied(OccupationNumber):
    def __init__(
        self,
        index: str,
        spin: Union[Literal["↑"], Literal["↓"]],
    ):
        super().__init__(index=index, spin=spin)

    def text_representation(self) -> str:
        return f"n({self.spin},{self.index})"


class UnOccupied(OccupationNumber):
    def __init__(
        self,
        index: str,
        spin: Union[Literal["↑"], Literal["↓"]],
    ):
        super().__init__(index=index, spin=spin)

    def text_representation(self) -> str:
        return f"[1-n({self.spin},{self.index})]"


def operators() -> Dict[str, List[OccupationNumber]]:
    return {
        "ClCHm": [UnOccupied("l", UP), Occupied("m", UP)],  # O1
        "DlDHm": [UnOccupied("l", DOWN), Occupied("m", DOWN)],  # O2
        "ClCmCHlCHmDlDHm": [  # O3
            UnOccupied("l", UP),
            UnOccupied("m", UP),
            UnOccupied("l", DOWN),
            Occupied("m", DOWN),
        ],
        "ClCHmDlDmDHlDHm": [  # O4
            UnOccupied("l", DOWN),
            UnOccupied("m", DOWN),
            UnOccupied("l", UP),
            Occupied("m", UP),
        ],
        "ClCHlDlDHm": [  # O7
            UnOccupied("l", UP),
            UnOccupied("l", DOWN),
            Occupied("m", DOWN),
        ],
        "CmCHmDlDHm": [
            UnOccupied("m", UP),
            UnOccupied("l", DOWN),
            Occupied("m", DOWN),
        ],  # O8
        "ClCHmDlDHl": [
            UnOccupied("l", UP),
            Occupied("m", UP),
            UnOccupied("l", DOWN),
        ],  # O5
        "ClCHmDmDHm": [
            UnOccupied("l", UP),
            Occupied("m", UP),
            UnOccupied("m", DOWN),
        ],  # O6
    }


def replace_index_where_spin(
    op: List[OccupationNumber],
    index_from: str,
    index_to: str,
    spin_from: Union[Literal["↑"], Literal["↓"]],
    spin_to: Union[Literal["↑"], Literal["↓"]],
) -> List[OccupationNumber]:
    mutable_copy = deepcopy(op)

    for elem in mutable_copy:
        if elem.spin == spin_from and elem.index == index_from:
            elem.overwrite_index(index_to)
            elem.overwrite_spin(spin_to)

    return mutable_copy


def join_op(op: List[OccupationNumber]) -> str:
    return "*".join([f"{asd.text_representation()}" for asd in op])


def print_difference(
    op: List[OccupationNumber],
):
    arr: List[Union[Literal["↑"], Literal["↓"]]] = [UP, DOWN]
    for swap_spin_a in arr:
        for swap_spin_b in arr:

            out_arr: List[str] = []

            for replace_l_with, replace_m_with in [
                ("j", "i"),  # i  = m    j  = l
                ("i", "j"),  # i  = l    j  = m
                ("l", "i"),  # i  = m    j != l
                ("l", "j"),  # i != l    j  = m
                ("j", "m"),  # i != m    j  = l
                ("i", "m"),  # i  = l    j != m
            ]:
                lam = "Λ(l,m)".replace("l", replace_l_with).replace("m", replace_m_with)
                sum = "           "
                if not (replace_l_with != "l" and replace_m_with != "m"):
                    sum = f"sum_nb({replace_l_with},{replace_m_with})"

                chain = op
                chain = replace_index_where_spin(chain, "l", replace_l_with, UP, UP)
                chain = replace_index_where_spin(chain, "m", replace_m_with, UP, UP)
                chain = replace_index_where_spin(chain, "l", replace_l_with, DOWN, DOWN)
                chain = replace_index_where_spin(chain, "m", replace_m_with, DOWN, DOWN)

                left_side = join_op(chain)

                ij_swapped_chain = chain
                ij_swapped_chain = replace_index_where_spin(
                    ij_swapped_chain, "i", "TEMP", swap_spin_a, swap_spin_b
                )
                ij_swapped_chain = replace_index_where_spin(
                    ij_swapped_chain, "j", "i", swap_spin_b, swap_spin_a
                )
                ij_swapped_chain = replace_index_where_spin(
                    ij_swapped_chain, "TEMP", "j", UP, UP
                )
                ij_swapped_chain = replace_index_where_spin(
                    ij_swapped_chain, "TEMP", "j", DOWN, DOWN
                )

                right_side = join_op(ij_swapped_chain)

                if left_side != right_side:
                    out_arr.append(f"    {sum} {lam} {{{left_side} - {right_side}}}")

            if len(out_arr):
                print(f"  Swap: n({swap_spin_a},i) <-> n({swap_spin_b},j)")
                for out in out_arr:
                    print(out)
                print()


if __name__ == "__main__":
    ops = operators()

    for key in ops.keys():
        print(f"Part Operator: {key}")
        print_difference(ops[key])
        print(f"")
        print(f"")
