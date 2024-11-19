from typing import List, Callable, Tuple
from common import (
    indent,
    generate_if_tree,
    write_file,
    init_file,
    is_basically_one,
    is_basically_zero,
)


FILENAME = "./../../computation-scripts/vcomponentssecondorder.py"


def generateHelperFile(inputMappings):
    init_file(FILENAME)

    # normal V
    write_file(
        FILENAME,
        "def v_second(U: float, E: float, t: float, knows_l_array: List[Tuple[int, float, int, float, int, float, int, float]], system_state, flipping_tuples: List[Tuple[int, bool]] = []) -> np.complex128:\n",
    )
    write_file(FILENAME, indent(1) + "res: np.complex128 = np.complex128(0)\n")
    write_file(
        FILENAME,
        indent(1) + "for (l, epsl, m, epsm, a, epsa, b, epsb) in knows_l_array:\n",
    )
    write_file(
        FILENAME,
        indent(2)
        + "eps_one_A = E*(epsl-epsm)\n"
        + indent(2)
        + "eps_one_B = eps_one_A + U\n"
        + indent(2)
        + "eps_one_C = eps_one_A - U\n"
        + indent(2)
        + "eps_two_A = E*(epsa-epsb)\n"
        + indent(2)
        + "eps_two_B = eps_two_A + U\n"
        + indent(2)
        + "eps_two_C = eps_two_A - U\n\n",
    )
    write_file(
        FILENAME,
        indent(2)
        + "occ_l = system_state.get_state_array()[l]\n"
        + indent(2)
        + "occ_l_os = system_state.get_state_array()[system_state.get_opposite_spin_index(l)]\n"
        + indent(2)
        + "occ_m = system_state.get_state_array()[m]\n"
        + indent(2)
        + "occ_m_os = system_state.get_state_array()[system_state.get_opposite_spin_index(m)]\n"
        + indent(2)
        + "occ_a = system_state.get_state_array()[a]\n"
        + indent(2)
        + "occ_a_os = system_state.get_state_array()[system_state.get_opposite_spin_index(a)]\n"
        + indent(2)
        + "occ_b = system_state.get_state_array()[b]\n"
        + indent(2)
        + "occ_b_os = system_state.get_state_array()[system_state.get_opposite_spin_index(b)]\n\n",
    )
    # This is in fact slower, as two array writes before and two after... That is so sad... # TODO complain in thesis
    write_file(
        FILENAME,
        indent(2)
        + "for flip_index, flip_up in flipping_tuples:\n"
        + indent(3)
        + "if flip_up:\n"
        + indent(4)
        + "if flip_index == l:\n"
        + indent(5)
        + "occ_l = 1 - occ_l\n"
        + indent(4)
        + "if flip_index == m:\n"
        + indent(5)
        + "occ_m = 1 - occ_m\n"
        + indent(4)
        + "if flip_index == a:\n"
        + indent(5)
        + "occ_a = 1 - occ_a\n"
        + indent(4)
        + "if flip_index == b:\n"
        + indent(5)
        + "occ_b = 1 - occ_b\n"
        + indent(3)
        + "else:\n"
        + indent(4)
        + "if flip_index == l:\n"
        + indent(5)
        + "occ_l_os = 1 - occ_l_os\n"
        + indent(4)
        + "if flip_index == m:\n"
        + indent(5)
        + "occ_m_os = 1 - occ_m_os\n"
        + indent(4)
        + "if flip_index == a:\n"
        + indent(5)
        + "occ_a_os = 1 - occ_a_os\n"
        + indent(4)
        + "if flip_index == b:\n"
        + indent(5)
        + "occ_b_os = 1 - occ_b_os\n\n",
    )

    def endCallback(lineStart: str, currentTruthinesses: List[bool]):
        l_is_a, l_is_b, m_is_a, m_is_b, Lc, Ld, Mc, Md, Ac, Ad, Bc, Bd = (
            currentTruthinesses
        )

        res = ""
        res += (
            lineStart
            + f"# l_is_a: {l_is_a} l_is_b: {l_is_b} m_is_a: {m_is_a} m_is_b: {m_is_b} Lc: {Lc} Ld: {Ld} Mc: {Mc} Md: {Md} Ac: {Ac} Ad: {Ad} Bc: {Bc} Bd: {Bd} "
            + "\n"
        )
        res += lineStart + "res += 0 "
        for (
            factor_t1_greater,
            factor_t2_greater,
            factor_sum,
            (mapping_t1_greater, mapping_t2_greater, mapping_first_order_product),
        ) in inputMappings.values():
            # integral part 1
            mult_t1 = mapping_t1_greater(
                l_is_a, l_is_b, m_is_a, m_is_b, Lc, Ld, Mc, Md, Ac, Ad, Bc, Bd
            )
            if not is_basically_zero(mult_t1):
                if is_basically_one(mult_t1):
                    res += "+" + factor_t1_greater
                elif is_basically_one(-mult_t1):
                    res += "-" + factor_t1_greater
                else:
                    res += "+ " + str(mult_t1) + " * " + factor_t1_greater
            # integral part 2
            mult_t2 = mapping_t2_greater(
                l_is_a, l_is_b, m_is_a, m_is_b, Ac, Ad, Bc, Bd, Lc, Ld, Mc, Md
            )
            if not is_basically_zero(mult_t2):
                if is_basically_one(mult_t2):
                    res += "+" + factor_t2_greater
                elif is_basically_one(-mult_t2):
                    res += "-" + factor_t2_greater
                else:
                    res += "+ " + str(mult_t2) + " * " + factor_t2_greater
            # minus factor product
            mult_prod = -mapping_first_order_product(Lc, Ld, Mc, Md, Ac, Ad, Bc, Bd)
            if not is_basically_zero(mult_prod):
                if is_basically_one(mult_prod):
                    res += "+" + factor_sum
                elif is_basically_one(-mult_prod):
                    res += "-" + factor_sum
                else:
                    res += "+ " + str(mult_prod) + " * " + factor_sum
        res += "\n"
        return res

    write_file(
        FILENAME,
        generate_if_tree(
            2,
            [
                "l == a",
                "l == b",
                "m == a",
                "m == b",
                "occ_l",
                "occ_l_os",
                "occ_m",
                "occ_m_os",
                "occ_a",
                "occ_a_os",
                "occ_b",
                "occ_b_os",
            ],
            endCallback,
        ),
    )

    write_file(FILENAME, indent(1) + "return res\n\n\n")


if __name__ == "__main__":

    def combine_funcs(eps_one, eps_two):
        return f"((np.expm1(1j * {eps_one} * t) / ({eps_one} * {eps_two})) - ( ((1j * t)/({eps_two}))   if (({eps_one} + {eps_two})< 1e-8) else    (np.expm1(1j * ({eps_one} + {eps_two}) * t) / ({eps_two} * ({eps_one} + {eps_two})))    )      )"

    def combine_funcs_reverse(eps_one, eps_two):
        return f"((np.exp(1j * {eps_two} * t) * np.expm1(1j * {eps_one} * t) / (-1 * {eps_one} * {eps_two})) + ( ((1j * t)/({eps_two}))   if (({eps_one} + {eps_two})< 1e-8) else    (np.expm1(1j * ({eps_one} + {eps_two}) * t) / ({eps_two} * ({eps_one} + {eps_two})))    )      )"

    def combine_funcs_sum(eps_one, eps_two):
        return f"((np.expm1(1j * {eps_one} * t) * np.expm1(1j * {eps_two} * t)) / ((-1 * {eps_one} * {eps_two})))"

    def double_eval_wrapper(
        callback_1_c: Callable[[int, int, int, int], int],
        callback_1_d: Callable[[int, int, int, int], int],
        callback_2_c: Callable[[int, int, int, int], int],
        callback_2_d: Callable[[int, int, int, int], int],
    ) -> Callable[
        [bool, bool, bool, bool, int, int, int, int, int, int, int, int], int
    ]:
        def temp_fun(
            l_is_a,
            l_is_b,
            m_is_a,
            m_is_b,
            occ_l,
            occ_l_os,
            occ_m,
            occ_m_os,
            occ_a,
            occ_a_os,
            occ_b,
            occ_b_os,
        ) -> int:
            tuples = [
                (True, callback_1_c, callback_2_c),
                (True, callback_1_c, callback_2_d),
                (False, callback_1_d, callback_2_c),
                (False, callback_1_d, callback_2_d),
            ]

            res = 0

            for flips_c, first_callback, second_callback in tuples:
                sub_res_first = first_callback(occ_l, occ_l_os, occ_m, occ_m_os)

                input_2_a = occ_a
                input_2_a_os = occ_a_os
                input_2_b = occ_b
                input_2_b_os = occ_b_os

                if flips_c:
                    if l_is_a:
                        # a up is now different
                        input_2_a = 1 - occ_l
                    if l_is_b:
                        # b up is now different
                        input_2_b = 1 - occ_l
                    if m_is_a:
                        # a up is now different
                        input_2_a = 1 - occ_m
                    if m_is_b:
                        # b up is now different
                        input_2_b = 1 - occ_m
                else:
                    if l_is_a:
                        # a down is now different
                        input_2_a_os = 1 - occ_l_os
                    if l_is_b:
                        # b down is now different
                        input_2_b_os = 1 - occ_l_os
                    if m_is_a:
                        # a down is now different
                        input_2_a_os = 1 - occ_m_os
                    if m_is_b:
                        # b down is now different
                        input_2_b_os = 1 - occ_m_os

                sub_res_second = second_callback(
                    input_2_a, input_2_a_os, input_2_b, input_2_b_os
                )

                res += sub_res_first * sub_res_second

            return res

        return temp_fun

    def eval_wrapper_packer(
        callback_1_c: Callable[[int, int, int, int], int],
        callback_1_d: Callable[[int, int, int, int], int],
        callback_2_c: Callable[[int, int, int, int], int],
        callback_2_d: Callable[[int, int, int, int], int],
    ) -> Tuple[
        # <N|Fi(l,m)Fj(a,b)|K> # inputs l_is_a, l_is_b, m_is_a, m_is_b,     occ_l, occ_l_os, occ_m, occ_m_os,     occ_a, occ_a_os, occ_b, occ_b_os
        Callable[[bool, bool, bool, bool, int, int, int, int, int, int, int, int], int],
        # <N|Fj(a,b)Fi(l,m)|K> # inputs l_is_a, l_is_b, m_is_a, m_is_b,     occ_a, occ_a_os, occ_b, occ_b_os,     occ_l, occ_l_os, occ_m, occ_m_os
        Callable[[bool, bool, bool, bool, int, int, int, int, int, int, int, int], int],
        # <N|Fi(l,m)|M><N|Fj(a,b)|L> # inputs occ_l, occ_l_os, occ_m, occ_m_os,     occ_a, occ_a_os, occ_b, occ_b_os
        Callable[[int, int, int, int, int, int, int, int], int],
    ]:
        def productOfFirstOrderCallable(
            occ_l,
            occ_l_os,
            occ_m,
            occ_m_os,
            occ_a,
            occ_a_os,
            occ_b,
            occ_b_os,
        ):
            return (
                callback_1_c(
                    occ_l,
                    occ_l_os,
                    occ_m,
                    occ_m_os,
                )
                + callback_1_d(
                    occ_l,
                    occ_l_os,
                    occ_m,
                    occ_m_os,
                )
            ) * (
                callback_2_c(
                    occ_a,
                    occ_a_os,
                    occ_b,
                    occ_b_os,
                )
                + callback_2_d(
                    occ_a,
                    occ_a_os,
                    occ_b,
                    occ_b_os,
                )
            )

        return (
            double_eval_wrapper(
                callback_1_c,
                callback_1_d,
                callback_2_c,
                callback_2_d,
            ),
            double_eval_wrapper(
                callback_2_c,
                callback_2_d,
                callback_1_c,
                callback_1_d,
            ),
            productOfFirstOrderCallable,
        )

    def A_fun_c(Lc, Ld, Mc, Md):
        return Lc * (1 - Mc) * (1 * (Ld == Md))

    def B_fun_c(Lc, Ld, Mc, Md):
        return Lc * (1 - Mc) * Ld * (1 - Md)

    def C_fun_c(Lc, Ld, Mc, Md):
        return Lc * (1 - Mc) * Md * (1 - Ld)

    def A_fun_d(Lc, Ld, Mc, Md):
        return Ld * (1 - Md) * (1 * (Lc == Mc))

    def B_fun_d(Lc, Ld, Mc, Md):
        return Ld * (1 - Md) * Lc * (1 - Mc)

    def C_fun_d(Lc, Ld, Mc, Md):
        return Ld * (1 - Md) * Mc * (1 - Lc)

    mappingsDict = {
        "AA": (
            combine_funcs("eps_one_A", "eps_two_A"),
            combine_funcs_reverse("eps_two_A", "eps_one_A"),
            combine_funcs_sum("eps_two_A", "eps_one_A"),
            eval_wrapper_packer(A_fun_c, A_fun_d, A_fun_c, A_fun_d),
        ),
        "AB": (
            combine_funcs("eps_one_A", "eps_two_B"),
            combine_funcs_reverse("eps_one_A", "eps_two_B"),
            combine_funcs_sum("eps_one_A", "eps_two_B"),
            eval_wrapper_packer(A_fun_c, A_fun_d, B_fun_c, B_fun_d),
        ),
        "AC": (
            combine_funcs("eps_one_A", "eps_two_C"),
            combine_funcs_reverse("eps_one_A", "eps_two_C"),
            combine_funcs_sum("eps_one_A", "eps_two_C"),
            eval_wrapper_packer(A_fun_c, A_fun_d, C_fun_c, C_fun_d),
        ),
        "BA": (
            combine_funcs("eps_one_B", "eps_two_A"),
            combine_funcs_reverse("eps_one_B", "eps_two_A"),
            combine_funcs_sum("eps_one_B", "eps_two_A"),
            eval_wrapper_packer(B_fun_c, B_fun_d, A_fun_c, A_fun_d),
        ),
        "BB": (
            combine_funcs("eps_one_B", "eps_two_B"),
            combine_funcs_reverse("eps_one_B", "eps_two_B"),
            combine_funcs_sum("eps_one_B", "eps_two_B"),
            eval_wrapper_packer(B_fun_c, B_fun_d, B_fun_c, B_fun_d),
        ),
        "BC": (
            combine_funcs("eps_one_B", "eps_two_C"),
            combine_funcs_reverse("eps_one_B", "eps_two_C"),
            combine_funcs_sum("eps_one_B", "eps_two_C"),
            eval_wrapper_packer(B_fun_c, B_fun_d, C_fun_c, C_fun_d),
        ),
        "CA": (
            combine_funcs("eps_one_C", "eps_two_A"),
            combine_funcs_reverse("eps_one_C", "eps_two_A"),
            combine_funcs_sum("eps_one_C", "eps_two_A"),
            eval_wrapper_packer(C_fun_c, C_fun_d, A_fun_c, A_fun_d),
        ),
        "CB": (
            combine_funcs("eps_one_C", "eps_two_B"),
            combine_funcs_reverse("eps_one_C", "eps_two_B"),
            combine_funcs_sum("eps_one_C", "eps_two_B"),
            eval_wrapper_packer(C_fun_c, C_fun_d, B_fun_c, B_fun_d),
        ),
        "CC": (
            combine_funcs("eps_one_C", "eps_two_C"),
            combine_funcs_reverse("eps_one_C", "eps_two_C"),
            combine_funcs_sum("eps_one_C", "eps_two_C"),
            eval_wrapper_packer(C_fun_c, C_fun_d, C_fun_c, C_fun_d),
        ),
    }

    generateHelperFile(mappingsDict)
