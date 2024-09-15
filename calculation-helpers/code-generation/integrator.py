def int_sin_sin(a: str, b: str):
    return f"( ( (({b})-({a})) * np.sin( (({b})+({a})) * t ) + (({b})+({a})) * np.sin( (({a})-({b})) * t ) ) /(2 *(({a})**2 - ({b})**2)) )"


def int_e_sin(a: str, b: str):
    return f"( ( np.exp(({a}) * t) * ( ({a}) * np.sin(({b}) * t) - ({b}) * np.cos(({a}) * t) ) + ({b}) ) / (({a})**2 + ({b})**2) )"


if __name__ == "__main__":
    print("16 CF")

    print()

    print("(", end="")
    print(int_sin_sin("U + epsm", "U + epsl"), end="")
    print("-", end="")
    print(int_sin_sin("epsm", "U + epsl"), end="")
    print("-", end="")
    print(int_sin_sin("U + epsm", "epsl"), end="")
    print("+", end="")
    print(int_sin_sin("epsm", "epsl"), end="")
    print(")", end="")

    print()
    print()
    print()

    print("2 i AE")

    print()

    print("(", end="")
    print(int_e_sin("-1j * epsm", "epsl"), end="")
    print(")", end="")

    print()
    print()
    print()

    print("2 i BD")

    print()
    print()
    print()

    print("(", end="")
    print(int_e_sin("1j * epsl", "epsm"), end="")
    print(")", end="")

    print()
    print()
    print()

    print("8 BF")

    print()

    print("(", end="")
    print(int_sin_sin("epsm", "U + epsl"), end="")
    print("-", end="")
    print(int_sin_sin("epsm", "epsl"), end="")
    print(")", end="")

    print()
    print()
    print()

    print("8 CE")

    print()

    print("(", end="")
    print(int_sin_sin("epsl", "U + epsm"), end="")
    print("-", end="")
    print(int_sin_sin("epsl", "epsm"), end="")
    print(")", end="")

    print()
    print()
    print()

    print("4 i AF")

    print()

    print("(", end="")
    print(int_e_sin("-1j * epsm", "U + epsl"), end="")
    print("-", end="")
    print(int_e_sin("-1j * epsm", "epsl"), end="")
    print(")", end="")

    print()
    print()
    print()

    print("4 BE")

    print()

    print("(", end="")
    print(int_sin_sin("epsl", "epsm"), end="")
    print(")", end="")

    print()
    print()
    print()

    print("AD")

    print()

    print("(", end="")
    print("((1j * np.exp(1j * (epsl - epsm) * t)) - 1) / (epsm-epsl)", end="")
    print(")", end="")

    print()
