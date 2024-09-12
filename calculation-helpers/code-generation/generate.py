import subprocess


def runCommandWithInput(command, input_string):
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:

        stdout, stderr = process.communicate(input_string)

        if process.returncode != 0:
            return f"Error: {stderr}"

    return stdout


def execEspressoOnInput(inputStr: str):
    return runCommandWithInput(["./program-generation/bin/espresso"], inputStr)


def xor(a, b):
    return bool(a) ^ bool(b)


def genEspressoInput(logicCallback, restrict=True):
    result = ""

    result += ".i 4\n"  # inputs
    result += ".o 1\n"  # outputs
    result += f".p {2**4}\n"  # number of terms in the file following

    for Lc in range(2):
        for Mc in range(2):
            for Ld in range(2):
                for Md in range(2):
                    res = logicCallback(Lc, Mc, Ld, Md)
                    if restrict:
                        if res > 1:
                            raise Exception("Callback greater 1")
                        result += f"{Lc}{Mc}{Ld}{Md} {1 if res else 0}\n"
                    else:
                        result += f"{Lc}{Mc}{Ld}{Md} {res}\n"

    result += ".e"
    return result


def printDifference(logicCallback, logicCallback2):
    for Lc in range(2):
        for Mc in range(2):
            for Ld in range(2):
                for Md in range(2):
                    a = logicCallback(Lc, Mc, Ld, Md)
                    if a == True:
                        a = 1
                    if a == False:
                        a = 0

                    b = logicCallback2(Lc, Mc, Ld, Md)
                    if b == True:
                        b = 1
                    if b == False:
                        b = 0

                    if a != b:
                        print(f"{Lc}{Mc}{Ld}{Md}   {a} <-> {b}")


if __name__ == "__main__":

    mappingsDict = {
        "16CF": [
            lambda Lc, Mc, Ld, Md: Lc * Ld * Md * (1 - Mc) + Ld * Lc * Mc * (1 - Md),
            lambda Lc, Mc, Ld, Md: Lc * Ld * (Md + Mc - (2 * Mc * Md)),
            lambda Lc, Mc, Ld, Md: Lc and Ld and xor(Md, Mc),
        ],
        "2iAE": [
            lambda Lc, Mc, Ld, Md: Lc
            - (Lc * (1 - Mc) * Ld)
            + Ld
            - (Ld * (1 - Md) * Lc),
            # lambda Lc, Mc, Ld, Md: Lc != Mc or Mc and Ld or Md and Ld, # doesn't capture 1111 being 2
        ],
        "2iBD": [
            lambda Lc, Mc, Ld, Md: Lc * (1 - Mc) * Md + Ld * (1 - Md) * Mc,
        ],
        "8BF": [
            lambda Lc, Mc, Ld, Md: Lc * Ld * (1 - Mc - Md) + Ld * Lc * (1 - Md - Mc),
        ],
        "8CE": [
            lambda Lc, Mc, Ld, Md: Lc * Md * (1 - Mc) + Ld * Mc * (1 - Md),
        ],
        "4iAF": [
            lambda Lc, Mc, Ld, Md: Lc * Ld + Ld * Lc,
            lambda Lc, Mc, Ld, Md: 2 * Lc * Ld,
        ],
        "4BE": [
            lambda Lc, Mc, Ld, Md: Lc * (1 - Mc - Md)
            + Lc * (1 - Mc) * Ld * Md
            + Ld * (1 - Md - Mc)
            + Ld * (1 - Md) * Lc * Mc,
        ],
        "AD": [lambda Lc, Mc, Ld, Md: Lc * (1 - Mc) + Ld * (1 - Md)],
    }

    for key, mappings in mappingsDict.items():
        print("\n\n\n" + key)

        canEspresso = True
        try:
            test = genEspressoInput(mappings[0], canEspresso)
        except Exception:
            canEspresso = False
            test = genEspressoInput(mappings[0], False)  # CAN NO LONGER ERROR

        print(test)

        if canEspresso:
            print(execEspressoOnInput(test))

        for mapping in mappings:
            if genEspressoInput(mapping, canEspresso) != test:
                print("Difference:")
                printDifference(mappings[0], mapping)

                raise Exception(f"Difference found at {key}")

    print("Lc Mc Ld Md")
