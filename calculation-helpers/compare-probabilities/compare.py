def compare_files(file1, file2):
    sum_1 = 0
    sum_2 = 0
    diffs = []

    with open(file1, "r") as f1, open(file2, "r") as f2:
        # Skip the first line in both files (name)
        next(f1)
        next(f2)

        while True:
            try:
                state1 = next(f1).strip()
                value1 = float(next(f1).strip())

                state2 = next(f2).strip()
                value2 = float(next(f2).strip())

                sum_1 += value1
                sum_2 += value2

                # Compare the states
                if state1 != state2:
                    print(f"States don't match: {state1} vs {state2}")
                    return

                print(f"Matching state: {state1}")

                # Calculate and print the percentage difference
                percentage_diff = abs((value1 - value2) / ((value1 + value2) / 2)) * 100
                print(f"Percentage difference: {percentage_diff:.2f}%")

                diffs.append((percentage_diff, state1, value1, value2))

            except StopIteration:
                break

    print()
    print("Normalization")
    print(file1)
    print(sum_1)
    print(file2)
    print(sum_2)

    print()
    print("Sorted")
    sorted_diffs = sorted(diffs, key=lambda x: x[0])
    for percentage_diff, state, val1, val2 in sorted_diffs:
        print(f"{state}: {percentage_diff:.2f}%  -  {val1:.6f} <-> {val2:.6f}")
    print(f"{file1} <-> {file2}")


if __name__ == "__main__":
    # at time 390 * 1/J, n=4
    compare_files("diagonalization.txt", "perturbation.txt")
