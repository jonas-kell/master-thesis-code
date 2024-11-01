from typing import List


def is_basically_one(num) -> bool:
    return abs(1 - num) < 1e-6


def is_basically_zero(num) -> bool:
    return abs(num) < 1e-6


# This function marks the GENERATED FILE as not to be edited manually
def init_file(FILENAME: str):
    text = """# !! THIS FILE IS AUTOMATICALLY GENERATED.
# DO NOT EDIT. 
# SEE calculation-helpers/code-generation/generate*.py

from typing import List, Tuple
import numpy as np

"""
    with open(FILENAME, "w") as file:
        file.write(text)


def write_file(FILENAME: str, input: str):
    with open(FILENAME, "a") as file:
        file.write(input)


def indent(spaces: int):
    return "    " * spaces


def generate_if_tree(
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
        res += generate_if_tree(
            initialIndent,
            checks,
            callbackForTop,
            currentDepth + 1,
            currentTruthinesses + [True],
        )
        res += indent(initialIndent + currentDepth) + "else:\n"
        res += generate_if_tree(
            initialIndent,
            checks,
            callbackForTop,
            currentDepth + 1,
            currentTruthinesses + [False],
        )
        return res