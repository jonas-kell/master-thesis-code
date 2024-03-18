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
OCCUPATION_NUMBER_FUNCTION = Function("n", positive=True)  # type: ignore

FILENAME = "./../spin-onehalf-square-hcbosons/analyticalcalcfunctions.py"


def init_file():
    text = """# !! THIS FILE IS AUTOMATICALLY GENERATED.
# DO NOT EDIT. 
# SEE simplificationtermhelper.py

from typing import Callable, List, Tuple

"""

    with open(FILENAME, "w") as file:
        file.write(text)


def write_file(input: str):
    with open(FILENAME, "a") as file:
        file.write(input)


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


def custom_printer(expr: Any, print_to_file: bool, top_level: bool = True) -> str:
    if expr.is_Add:
        positive_terms: List[str] = []
        negative_terms: List[str] = []
        for term in expr.args:
            if term.is_negative:
                negative_terms.append(custom_printer(term, print_to_file, False))
            else:
                positive_terms.append(custom_printer(term, print_to_file, False))
        out = ""
        if not top_level:
            out += "("
        out += "+".join(positive_terms) + "".join(negative_terms)
        if not top_level:
            out += ")"
        return out
    elif expr.is_Mul:
        factor_strings: List[str] = []
        negative = False
        for factor in expr.args:
            if factor.is_negative:
                negative = True
                if factor != -1:
                    factor_strings.append(custom_printer(-factor, print_to_file, False))
            else:
                factor_strings.append(custom_printer(factor, print_to_file, False))

        out = ""
        if negative:
            out += "-"
        if len(factor_strings) > 1 and not top_level:
            out += "("
        out += "*".join(factor_strings)
        if len(factor_strings) > 1 and not top_level:
            out += ")"
        return out
    elif expr.is_Function:
        if print_to_file:
            # name = expr.name  # always the same, only occupation n
            # spin = str(expr.args[0].name) # should be automatically encoded in index, convenient
            index = str(expr.args[1].name)

            if index == "i":
                return "sw1_occupation"
            if index == "j":
                return "sw2_occupation"
            return "nb_occupation"
        else:
            return str(expr)
    else:
        return str(expr)


def print_difference(
    name: str,
    op: List[OccupationNumber],
    simplify_output: bool,
    print_to_file: bool,
):
    if print_to_file:
        write_file(
            f"def {name}(sw1_up: bool, sw1_index: int, sw1_occupation: int, sw2_up: bool, sw2_index: int, sw2_occupation: int, lam: Callable[[int, int], float], sw1_neighbors_index_occupation_tuples: List[Tuple[int,int]], sw2_neighbors_index_occupation_tuples: List[Tuple[int,int]]) -> float:\n    res:float = 0\n"
        )

    arr: List[Union[Literal["↑"], Literal["↓"]]] = [UP, DOWN]
    for swap_spin_a in arr:
        for swap_spin_b in arr:

            if print_to_file:
                if swap_spin_a == UP and swap_spin_b == UP:
                    write_file(
                        f"    if sw1_up and sw2_up:\n        # UP<->UP\n        pass\n"
                    )

                if swap_spin_a == UP and swap_spin_b == DOWN:
                    write_file(
                        f"    if sw1_up and not sw2_up:\n        # UP<->DOWN\n        pass\n"
                    )

                if swap_spin_a == DOWN and swap_spin_b == UP:
                    write_file(
                        f"    if not sw1_up and sw2_up:\n        # DOWN<->UP\n        pass\n"
                    )

                if swap_spin_a == DOWN and swap_spin_b == DOWN:
                    write_file(
                        f"    if not sw1_up and not sw2_up:\n        # DOWN<->DOWN\n        pass\n"
                    )

            if print_to_file:
                write_file(f"")

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
                sum_arg_1 = ""
                sum_arg_2 = ""
                needs_sum = False
                if not (replace_l_with != "l" and replace_m_with != "m"):
                    if replace_l_with == "l":
                        # swap so that sums are nicer
                        sum_arg_1 = replace_m_with
                        sum_arg_2 = replace_l_with
                    else:
                        sum_arg_1 = replace_l_with
                        sum_arg_2 = replace_m_with

                    needs_sum = True
                    sum = f"sum_nb({sum_arg_1},{sum_arg_2})"

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
                    out_arr.append(
                        f"    {sum} {lam} {{{custom_printer(full_term, False)}}}"
                    )
                    if print_to_file:
                        lam = (
                            lam.replace("Λ", "lam")
                            .replace("i", "sw1_index")
                            .replace("j", "sw2_index")
                        )

                        if not needs_sum:
                            # no sum, only add
                            write_file(
                                f"        res += {lam} * ({custom_printer(full_term, True)})\n"
                            )
                        else:
                            # sum-iterator
                            write_file(
                                f"        # sum({sum_arg_1},{sum_arg_2})\n"
                                + f"        for ({sum_arg_2}, nb_occupation) in {'sw1_neighbors_index_occupation_tuples' if sum_arg_1 == 'i' else 'sw2_neighbors_index_occupation_tuples'}:\n"
                            )
                            write_file(
                                f"            res += {lam} * ({custom_printer(full_term, True)})\n"
                            )

            if len(out_arr):
                print(f"  Swap: n({swap_spin_a},i) <-> n({swap_spin_b},j)")
                for out in out_arr:
                    print(out)
                print()

    if print_to_file:
        write_file(f"    return res\n\n\n")


if __name__ == "__main__":
    ops = operators()

    print_to_file = True

    if print_to_file:
        init_file()

    for key in ops.keys():
        print(f"Part Operator: {key}")
        print_difference(key, ops[key], True, print_to_file)
        print(f"")
        print(f"")

# TODO     sum_nb(i,l) Λ(l,i) {(n(↑, l)-1)*(n(↓, l)-1)*(n(↑, j)-n(↑, i))*n(↓, i)} properly treat spins, NOT encoded like I thought
