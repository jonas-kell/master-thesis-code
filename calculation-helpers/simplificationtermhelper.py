from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Union, Any
from copy import deepcopy
from sympy import simplify, evaluate, Function, Symbol, Mul  # type: ignore
from sympy.core.sympify import sympify  # type: ignore
from functools import reduce

UP = "↑"
DOWN = "↓"
UP_SYMBOL = Symbol(UP)
DOWN_SYMBOL = Symbol(DOWN)
OCCUPATION_NUMBER_FUNCTION = Function("n")  # type: ignore


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

    def get_spin_symbol(self) -> Symbol:
        if self.spin == UP:
            return UP_SYMBOL
        return DOWN_SYMBOL

    def get_index_symbol(self) -> Symbol:
        return Symbol(self.index)

    @abstractmethod
    def get_sympy_repr(self) -> Any:
        pass


class Occupied(OccupationNumber):
    def __init__(
        self,
        index: str,
        spin: Union[Literal["↑"], Literal["↓"]],
    ):
        super().__init__(index=index, spin=spin)

    def text_representation(self) -> str:
        return f"n({self.spin},{self.index})"

    def get_sympy_repr(self) -> Any:
        return OCCUPATION_NUMBER_FUNCTION(
            self.get_spin_symbol(),
            self.get_index_symbol(),
        )  # type: ignore


class UnOccupied(OccupationNumber):
    def __init__(
        self,
        index: str,
        spin: Union[Literal["↑"], Literal["↓"]],
    ):
        super().__init__(index=index, spin=spin)

    def text_representation(self) -> str:
        return f"[1-n({self.spin},{self.index})]"

    def get_sympy_repr(self) -> Any:
        return sympify(1) - OCCUPATION_NUMBER_FUNCTION(
            self.get_spin_symbol(),
            self.get_index_symbol(),
        )  # type: ignore


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


def join_op(op: List[OccupationNumber]) -> Any:
    return reduce(lambda a, b: Mul(a, b, evaluate=False), map(lambda c: c.get_sympy_repr(), op))  # type: ignore


def print_difference(
    op: List[OccupationNumber],
    simplify_output: bool,
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

                left_side = join_op(chain)
                right_side = join_op(ij_swapped_chain)

                with evaluate(False):
                    full_term = left_side - right_side

                is_zero = str(left_side) == str(right_side)
                if simplify_output:
                    full_term = simplify(full_term)  # type: ignore
                    if full_term.is_zero == True:  # type: ignore
                        is_zero = True

                if not is_zero:
                    out_arr.append(f"    {sum} {lam} {{{full_term}}}")

            if len(out_arr):
                print(f"  Swap: n({swap_spin_a},i) <-> n({swap_spin_b},j)")
                for out in out_arr:
                    print(out)
                print()


if __name__ == "__main__":
    ops = operators()

    for key in ops.keys():
        print(f"Part Operator: {key}")
        print_difference(ops[key], True)
        print(f"")
        print(f"")
