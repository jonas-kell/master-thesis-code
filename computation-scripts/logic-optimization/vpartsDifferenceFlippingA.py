"""Compute all 16 possibilities"""

from typing import Literal

mode: Literal["logic", "print"] = "logic"

# could optimize this: https://github.com/classabbyamp/espresso-logic


if mode == "print":  # type: ignore
    print("Lu Mu Ld Md   flp   a b c d")
if mode == "logic":  # type: ignore
    print(".i 6")  # inputs
    print(".o 2")  # outputs
    print(f".p {2**6}")  # number of terms in the file following

for flip in range(4):
    for Lu in range(2):
        for Mu in range(2):
            for Ld in range(2):
                for Md in range(2):
                    Lub = 1 - Lu
                    Ldb = 1 - Ld
                    Mub = 1 - Mu
                    Mdb = 1 - Md

                    Lup = Lu
                    Ldp = Ld
                    Mup = Mu
                    Mdp = Md
                    Lubp = Lub
                    Ldbp = Ldb
                    Mubp = Mub
                    Mdbp = Mdb

                    if flip == 0:
                        flipname = "Lu"
                        Lup = 1 - Lup
                        Lubp = 1 - Lubp
                    if flip == 1:
                        flipname = "Ld"
                        Ldp = 1 - Ldp
                        Ldbp = 1 - Ldbp
                    if flip == 2:
                        flipname = "Mu"
                        Mup = 1 - Mup
                        Mubp = 1 - Mubp
                    if flip == 3:
                        flipname = "Md"
                        Mdp = 1 - Mdp
                        Mdbp = 1 - Mdbp

                    a = 1 if Lu and Mub and Ld == Md else 0
                    b = 1 if Lup and Mubp and Ldp == Mdp else 0
                    c = 1 if Ld and Mdb and Lu == Mu else 0
                    d = 1 if Ldp and Mdbp and Lup == Mup else 0

                    if mode == "print":  # type: ignore
                        print(f"{Lu}  {Ld}  {Mu}  {Md}    {flipname} ", end="")
                        print(f"   {a} {b} {c} {d}    {a-b+c-d}")
                    if mode == "logic":  # type: ignore
                        print(f"{Lu}{Ld}{Mu}{Md}{flip%2}{flip//2} {a+c}{b+d}")


if mode == "logic":  # type: ignore
    print(".e")
