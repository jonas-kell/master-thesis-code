import numpy as np

U = 1.0
E = 0.8
J = 1.3
phi = np.pi * 0.6

time = 0.6

n = 2

nr_sites = 2 * n
nr_states = 2**nr_sites


def psia(state: np.array):
    res = 0

    for l in range(nr_sites // 2 - 1):
        m = l + 1
        los = l + nr_sites // 2
        mos = m + nr_sites // 2

        res += state[l] * (1 - state[m]) * (state[los] == state[mos])
        res += state[los] * (1 - state[mos]) * (state[l] == state[m])

    return res * J


def psib(state: np.array):
    res = 0

    for l in range(nr_sites // 2 - 1):
        l = l + 1
        m = l - 1
        los = l + nr_sites // 2
        mos = m + nr_sites // 2

        res += state[l] * (1 - state[m]) * (state[los] == state[mos])
        res += state[los] * (1 - state[mos]) * (state[l] == state[m])

    return res * J


def eps(index: int):
    return E * np.cos(phi) * ((index % (nr_sites // 2)) - (nr_sites // 2 - 1) / 2)


def energy_zero(state: np.array):
    res = np.complex128(0)

    for l in range(nr_sites // 2):
        res += state[l] * state[l + nr_sites // 2] * U

    for l in range(nr_sites):
        res += eps(l) * state[l]

    return res


def lama():
    return (np.exp(1j * (eps(0) - eps(1)) * time) - 1) / (eps(0) - eps(1))


def lamb():
    return (np.exp(1j * (eps(1) - eps(0)) * time) - 1) / (eps(1) - eps(0))


def hN(state: np.array):
    res = np.complex128(0)

    res += lama() * psia(state=state)
    res += lamb() * psib(state=state)

    return res


def heff(state: np.array):
    return -1j * energy_zero(state=state) * time + hN(state=state)


def main():
    for state_index in range(nr_states):
        state = np.zeros(nr_sites, dtype=np.int16)
        for state_shift_index in range(nr_sites):
            state[nr_sites - 1 - state_shift_index] = 1 - (
                (state_index & (1 << state_shift_index)) >> state_shift_index
            )

        print("state:", state)
        # start state computation

        # print("eps0", eps(0))
        # print("eps1", eps(1))

        # print("psia", psia(state=state))
        # print("psib", psib(state=state))
        # print("lam a", lama())
        # print("lam b", lamb())

        # print("E_0", energy_zero(state=state))
        # print("H_n", hN(state=state))
        print("heff", heff(state=state))

        print()


if __name__ == "__main__":
    main()
