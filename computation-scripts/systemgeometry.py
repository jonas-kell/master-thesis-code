from typing import Dict, Union, Any, List, Tuple
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
        self.index_knows_cache: List[
            List[Tuple[int, float, int, float, int, float, int, float]]
        ] = None

    def init_index_knows_cache(self, phi: float, sin_phi: float, cos_phi: float):
        # init index knows tuples
        self.index_knows_cache: List[
            List[Tuple[int, float, int, float, int, float, int, float]]
        ] = []
        for index in range(self.get_number_sites_wo_spin_degree()):
            inner_list: List[Tuple[int, float, int, float, int, float, int, float]] = []
            for nb_index in self.get_nearest_neighbor_indices(index):
                ab_indices: List[Tuple[int, int]] = []

                # a = l, b is nb of l
                for b in self.get_nearest_neighbor_indices(index):
                    ab_indices.append((index, b))
                # a = m, b is nb of m
                for b in self.get_nearest_neighbor_indices(nb_index):
                    ab_indices.append((nb_index, b))
                # b = l, a is nb of l & neq m
                for a in self.get_nearest_neighbor_indices(index):
                    if a != nb_index:
                        ab_indices.append((a, index))
                # b = m, a is nb of m & neq l
                for a in self.get_nearest_neighbor_indices(nb_index):
                    if a != index:
                        ab_indices.append((a, nb_index))

                for a, b in ab_indices:
                    inner_list.append(
                        (
                            index,
                            self.get_eps_multiplier(index, phi, sin_phi, cos_phi),
                            nb_index,
                            self.get_eps_multiplier(nb_index, phi, sin_phi, cos_phi),
                            a,
                            self.get_eps_multiplier(a, phi, sin_phi, cos_phi),
                            b,
                            self.get_eps_multiplier(b, phi, sin_phi, cos_phi),
                        )
                    )

            self.index_knows_cache.append(inner_list)

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

    def get_index_knows_tuples(
        self, index: int
    ) -> List[Tuple[int, float, int, float, int, float, int, float]]:
        if self.index_knows_cache is None:
            raise Exception(
                "Index knows cache is requested, but not yet initialized for this geometry"
            )
        return self.index_knows_cache[index]


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

        if False:
            # simple variant, index from left to right
            return cos_phi * ((index % domain_size))
        else:  # TODO Remove ?
            # Attempt to check the lateral symmetry of flipping, by placing the 0-index in the "middle" to make it E-symmetric
            # this should not influence other properties (I think)
            return cos_phi * ((index % domain_size) - (domain_size / 2.0) + 0.5)

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {"type": "LinearChainNonPeriodicState", "size": self.size}
