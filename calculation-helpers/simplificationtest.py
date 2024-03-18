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
