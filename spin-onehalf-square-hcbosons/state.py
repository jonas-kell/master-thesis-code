import numpy as np
from abc import ABC, abstractmethod
from typing import List
from randomgenerator import RandomGenerator


class SystemState(ABC):
    """
    expects storage in 1d-numpy-array

    First get_number_sites_wo_spin_degree() many entries
    ->     Spin Up
    Then once more  get_number_sites_wo_spin_degree() many entries
    ->     Spin Down
    """

    def __init__(self):
        self.state = np.zeros((self.get_number_sites(),), dtype=np.uint8)

    def get_state_array(self) -> np.ndarray:
        return self.state

    @abstractmethod
    def get_number_sites_wo_spin_degree(self) -> int:
        pass

    def get_number_sites(self) -> int:
        # two spin sites for every index
        return self.get_number_sites_wo_spin_degree() * 2

    def get_opposite_spin_index(self, index: int) -> int:
        return (
            index + self.get_number_sites_wo_spin_degree()
        ) % self.get_number_sites()

    @abstractmethod
    def get_nearest_neighbor_indices(self, index: int) -> List[int]:
        pass

    def init_random_filling(
        self, fill_ratio: float, random_generator: RandomGenerator
    ) -> None:
        if fill_ratio < 0:
            raise Exception("Fill ratio must be at least 0")

        if fill_ratio > 1:
            raise Exception("Fill ratio must be at most 1")

        all_sites = self.get_number_sites()
        target_num_filling = int(all_sites * fill_ratio)

        if fill_ratio <= 0.5:
            # init with zeros and add
            added = 0
            self.state = np.zeros_like(self.state)
            while added < target_num_filling:
                place_index = random_generator.randint(0, all_sites - 1)
                if self.state[place_index] == 0:
                    self.state[place_index] = 1
                    added += 1
        else:
            # init with ones and remove
            self.state = np.ones_like(self.state)
            removed = 0
            target_num_removing = all_sites - target_num_filling
            while removed < target_num_removing:
                place_index = random_generator.randint(0, all_sites - 1)
                if self.state[place_index] == 1:
                    self.state[place_index] = 0
                    removed += 1

    def get_random_flipped_copy(
        self, no_flips: int, random_generator: RandomGenerator
    ) -> np.ndarray:
        all_sites = self.get_number_sites()
        new_state = self.state.copy()

        for _ in range(no_flips):
            flip_index = random_generator.randint(0, all_sites - 1)

            new_state[flip_index] = (self.state[flip_index] + 1) % 2

        return new_state

    def set_state(self, new_state: np.ndarray) -> None:
        self.state = new_state

    @abstractmethod
    def get_eps_multiplier(
        self, index: int, phi: float, sin_phi: float, cos_phi: float
    ) -> float:
        pass

    @abstractmethod
    def get_Psi_of_N(self, system_state_array: np.ndarray) -> float:
        pass


# from typing import Type
# def scalar_product(self, mp_with: Type["SystemState"]) -> float:
#     return np.dot(self.state.ravel(), mp_with.state.ravel())


class SquareSystemNonPeriodicState(SystemState):
    """
    Will be numbered the following way (example for size M = 3)

     Spin Up
      0    1    2
      3    4    5
      6    7    8
     Spin Down
      9   10   11
     12   13   14
     15   16   17
    """

    def __init__(self, size: int):
        self.size = size
        super().__init__()

    def get_number_sites_wo_spin_degree(self) -> int:
        # square
        return self.size * self.size

    def get_nearest_neighbor_indices(self, index: int) -> List[int]:
        ## TODO: cache
        domain_size = self.get_number_sites_wo_spin_degree()
        M = self.size
        col_index = (index % domain_size) % M
        row_index = (index % domain_size) // M

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

    def get_eps_multiplier(
        self, index: int, phi: float, sin_phi: float, cos_phi: float
    ) -> float:
        phi  # get rid of unused error, sorry

        ## TODO: cache
        M = self.size
        domain_size = self.get_number_sites_wo_spin_degree()
        cut_index = index % domain_size
        return cos_phi * (cut_index % M) + sin_phi * (cut_index // M)

    def get_Psi_of_N(self, system_state_array: np.ndarray) -> float:
        system_state_array  # get rid of unused error, sorry

        return 1 / (2 ** self.get_number_sites_wo_spin_degree())


class LinearChainNonPeriodicState(SystemState):
    """
    Will be numbered the following way (example for size M = 3)

     Spin Up
      0    1    2
     Spin Down
      3    4    5
    """

    def __init__(self, size: int):
        self.size = size
        super().__init__()

    def get_number_sites_wo_spin_degree(self) -> int:
        return self.size

    def get_nearest_neighbor_indices(self, index: int) -> List[int]:
        domain_size = self.get_number_sites_wo_spin_degree()

        res = []

        # left neighbor
        if index > 0:
            res.append(index - 1)
        # right neighbor
        if index < domain_size - 1:
            res.append(index + 1)

        return res

    def get_eps_multiplier(
        self, index: int, phi: float, sin_phi: float, cos_phi: float
    ) -> float:
        phi  # get rid of unused error, sorry
        sin_phi

        domain_size = self.get_number_sites_wo_spin_degree()

        return cos_phi * (index % domain_size)

    def get_Psi_of_N(self, system_state_array: np.ndarray) -> float:
        system_state_array  # get rid of unused error, sorry

        return 1 / (2 ** self.get_number_sites_wo_spin_degree())
