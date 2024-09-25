# pip3 install sympy

#  type: ignore - file-global

from sympy import symbols, conjugate, Matrix, init_printing, sqrt
from sympy.core.sympify import sympify
from sympy.abc import x

init_printing(use_unicode=True)

# Define the expression
test = sympify(1j) * x - sympify(1j) * conjugate(x)

mat = Matrix([[test, test], [test, 2 * test]])

print(mat ** (1 / 2))

print(mat[3])
print(mat.shape)

mat2 = Matrix(
    [
        [symbols("a"), symbols("b"), symbols("c"), symbols("d")],
        [symbols("e"), symbols("f"), symbols("g"), symbols("h")],
        [symbols("k"), symbols("l"), symbols("m"), symbols("n")],
        [symbols("o"), symbols("p"), symbols("q"), symbols("r")],
    ]
)
# print(mat2.eigenvals())
print(mat2.trace())
print(sqrt(mat2))
