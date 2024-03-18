for a in range(2):
    for b in range(2):
        for c in range(2):
            for l in range(1, 4):
                for m in range(l - 1, l - 1 + 3):
                    print(
                        f"LA({l},{m}) <N|A({l},{m})|{a}{b}{c}> - LA({l},{m}) <N'|A({l},{m})|{a}{b}{c}>"
                    )
