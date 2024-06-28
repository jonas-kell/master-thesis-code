from typing import Dict, Union, Any, List
from abc import ABC, abstractmethod


class SystemGeometry(ABC):
    """
    expects storage in 1d-numpy-array

    First get_number_sites_wo_spin_degree() many entries
    ->     Spin Up
    Then once more  get_number_sites_wo_spin_degree() many entries
    ->     Spin Down
    """

    def __init__(self):
        pass

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

    @abstractmethod
    def get_eps_multiplier(
        self, index: int, phi: float, sin_phi: float, cos_phi: float
    ) -> float:
        pass

    @abstractmethod
    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        pass


class SquareSystemNonPeriodicState(SystemGeometry):
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

        res: List[int] = []

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
        _ = phi  # get rid of unused error

        ## TODO: cache
        M = self.size
        domain_size = self.get_number_sites_wo_spin_degree()
        cut_index = index % domain_size
        return cos_phi * (cut_index % M) + sin_phi * (cut_index // M)

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {"type": "SquareSystemNonPeriodicState", "size": self.size}


class LinearChainNonPeriodicState(SystemGeometry):
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

        res: List[int] = []
        rel_index = index % domain_size
        shift = (index // domain_size) * domain_size

        # left neighbor
        if rel_index > 0:
            res.append(rel_index - 1 + shift)
        # right neighbor
        if rel_index < domain_size - 1:
            res.append(rel_index + 1 + shift)

        return res

    def get_eps_multiplier(
        self, index: int, phi: float, sin_phi: float, cos_phi: float
    ) -> float:
        _ = phi
        _ = sin_phi  # get rid of unused error

        domain_size = self.get_number_sites_wo_spin_degree()

        return cos_phi * (index % domain_size)

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {"type": "LinearChainNonPeriodicState", "size": self.size}
