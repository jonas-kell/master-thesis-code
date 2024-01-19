import numpy as np
from abc import ABC, abstractmethod
from typing import Type
from typing import List


class SystemState(ABC):
    def __init__(self):
        self.state = np.zeros((self.get_number_sites(),), dtype=np.uint8)

    def get_state(self) -> np.ndarray:
        return self.state

    @abstractmethod
    def get_number_sites(self) -> int:
        pass

    @abstractmethod
    def get_neighbor_indices(self, index: int) -> List[int]:
        pass


# def scalar_product(self, mp_with: Type["SystemState"]) -> float:
#     return np.dot(self.state.ravel(), mp_with.state.ravel())


class SquareSystemNonPeriodicState(SystemState):
    # Will be numbered the following way (example for size M = 3)
    #
    #  Spin Up
    #   0    1    2
    #   3    4    5
    #   6    7    8
    #  Spin Down
    #   9   10   11
    #  12   13   14
    #  15   16   17
    #

    def __init__(self, size: int):
        self.size = size
        super().__init__()

    def get_number_sites(self) -> int:
        # square and two spin sites for every index
        return self.size * self.size * 2

    def get_nearest_neighbor_indices(self, index: int) -> List[int]:
        domain_size = self.size * self.size
        col_index = (index % domain_size) % self.size
        row_index = (index % domain_size) // self.size
        M = self.size

        res = []

        # top neighbor
        if row_index > 0:
            res.append(index - M)
        # left neighbor
        if col_index > 0:
            res.append(index - 1)
        # right neighbor
        if col_index < M - 1:
            res.append(index + 1)
        # bottom neighbor
        if row_index < M - 1:
            res.append(index + M)

        return res
