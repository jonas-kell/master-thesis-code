import subprocess


def run_command_with_input(command, input_string):
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
    return run_command_with_input(["./program-generation/bin/espresso"], inputStr)


def xor(a, b):
    return bool(a) ^ bool(b)


def genEspressoInput(logicCallback):
    result = ""

    result += ".i 4\n"  # inputs
    result += ".o 1\n"  # outputs
    result += f".p {2**6}\n"  # number of terms in the file following

    for Lc in range(2):
        for Mc in range(2):
            for Ld in range(2):
                for Md in range(2):
                    result += f"{Lc}{Mc}{Ld}{Md} {1 if logicCallback(Lc, Mc, Ld, Md) else 0}\n"

    result += ".e"
    return result


if __name__ == "__main__":

    mappingsDict = {
        "16CF": [
            lambda Lc, Mc, Ld, Md: Lc * Ld * (Md + Mc - (2 * Mc * Md)),
            lambda Lc, Mc, Ld, Md: Lc and Ld and xor(Md, Mc),
        ]
    }

    for key, mappings in mappingsDict.items():
        test = genEspressoInput(mappings[0])

        print("\n\n\n" + key)
        print(test)
        # print(execEspressoOnInput(test))

        for mapping in mappings:
            if genEspressoInput(mappings[0]) != test:
                raise Exception(f"Difference found at {key}")
