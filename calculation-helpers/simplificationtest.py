# pip3 install sympy

#  type: ignore - file-global

from sympy import simplify, evaluate, Function, Symbol
from sympy.core.sympify import sympify
from sympy.abc import x


# Define the expression
a = sympify(2) * x**2
b = x**2 + x
with evaluate(False):
    expression = a - 2 * b

# Simplify the expression
simplified_expression = simplify(expression)

# Print the simplified expression
print("Original expression:", expression)
print("Simplified expression:", simplified_expression)
print("Zero check:")
print(
    simplified_expression.is_zero == True
)  # if not determinable (because x could be x=0), returns None
print((simplified_expression + 2 * x).is_zero == True)

i = Symbol("i")
l = Symbol("l")
m = Symbol("m")
f = Function("f")

with evaluate(False):
    test1 = f(i, l, m) - f(i, l, Symbol("m"))
    test2 = f(i, l, m) - f(i, l, l)

print("test1 expression:", test1)
print("test1 expression:", simplify(test1))
print("test2 expression:", test2)
print("test2 expression:", simplify(test2))


print("")
print("")
print("Custom printing:")


from sympy import symbols, init_printing
from typing import List

init_printing(order="none")


def custom_printer(expr) -> str:
    if expr.is_Add:
        positive_terms = []
        negative_terms = []
        for term in expr.args:
            if term.is_negative:
                negative_terms.append(custom_printer(term))
            else:
                positive_terms.append(custom_printer(term))
        return "(" + "+".join(positive_terms) + "".join(negative_terms) + ")"
    elif expr.is_Mul:
        factor_strings: List[str] = []
        negative = False
        for factor in expr.args:
            if factor.is_negative:
                negative = True
                if factor != -1:
                    factor_strings.append(custom_printer(-factor))
            else:
                factor_strings.append(custom_printer(factor))

        out = ""
        if negative:
            out += "-"
        if len(factor_strings) > 1:
            out += "("
        out += "*".join(factor_strings)
        if len(factor_strings) > 1:
            out += ")"
        return out
    elif expr.is_Function:
        print(expr.name)
        print(expr.args)
        return str(expr)
    else:
        return str(expr)


# Define symbols
x, y, z = symbols("x y z", positive=True)

# Define expression
expr = x - y * (x * 2) + y * (-2 - x + 4 * Function("f")(x, y)) - 1

print(expr)
print(custom_printer(expr))
