from typing import Dict, Union, Any, List, Tuple, Set
from abc import ABC, abstractmethod
from time import time as measure


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
            Set[Tuple[int, float, int, float, int, float, int, float]]
        ] = None
        self.index_knows_cache_contains_one: List[
            Set[Tuple[int, float, int, float, int, float, int, float]]
        ] = None
        self.index_knows_cache_contains_two: List[
            List[Set[Tuple[int, float, int, float, int, float, int, float]]]
        ] = None
        self.index_overlap_circle_cache: Set[Tuple[int, int, int, int]] = None

    def init_index_overlap_circle_cache(self, circle_radius: int):
        time_start = measure() * 1000
        # init index overlap circle tuples
        if self.index_overlap_circle_cache is None:
            self.index_overlap_circle_cache: Set[Tuple[int, int, int, int]] = set()

            for index in range(self.get_number_sites_wo_spin_degree()):
                index_circle = set([index])
                for _ in range(circle_radius):
                    for check_index in index_circle.copy():
                        for plus_one_step_nb in self.get_nearest_neighbor_indices(
                            check_index
                        ):
                            index_circle.add(plus_one_step_nb)

                for l in index_circle:
                    for m in self.get_nearest_neighbor_indices(l):
                        for a in index_circle:
                            for b in self.get_nearest_neighbor_indices(a):
                                self.index_overlap_circle_cache.add((l, m, a, b))

        time_end = measure() * 1000
        print(
            f"Index-Overlap-Circle-Cache has been pre-computed in {time_end-time_start}ms to contain {len(self.index_overlap_circle_cache)}"
        )

    def init_index_knows_cache(self, phi: float, sin_phi: float, cos_phi: float):
        time_start = measure() * 1000
        # init index knows tuples
        if self.index_knows_cache is None:
            self.index_knows_cache: List[
                Set[Tuple[int, float, int, float, int, float, int, float]]
            ] = []
            for l in range(self.get_number_sites_wo_spin_degree()):
                inner_list: Set[
                    Tuple[int, float, int, float, int, float, int, float]
                ] = set()
                for m in self.get_nearest_neighbor_indices(l):
                    ab_indices: Set[Tuple[int, int]] = set()

                    # a = l, b is nb of l
                    for b in self.get_nearest_neighbor_indices(l):
                        ab_indices.add((l, b))
                    # a = m, b is nb of m
                    for b in self.get_nearest_neighbor_indices(m):
                        ab_indices.add((m, b))
                    # b = l, a is nb of l & neq m
                    for a in self.get_nearest_neighbor_indices(l):
                        if a != m:
                            ab_indices.add((a, l))
                    # b = m, a is nb of m & neq l
                    for a in self.get_nearest_neighbor_indices(m):
                        if a != l:
                            ab_indices.add((a, m))

                    for a, b in ab_indices:
                        inner_list.add(
                            (
                                l,
                                self.get_eps_multiplier(l, phi, sin_phi, cos_phi),
                                m,
                                self.get_eps_multiplier(m, phi, sin_phi, cos_phi),
                                a,
                                self.get_eps_multiplier(a, phi, sin_phi, cos_phi),
                                b,
                                self.get_eps_multiplier(b, phi, sin_phi, cos_phi),
                            )
                        )

                self.index_knows_cache.append(inner_list)

        # init index knows tuples that contain one index
        if self.index_knows_cache_contains_one is None:
            self.index_knows_cache_contains_one: List[
                Set[Tuple[int, float, int, float, int, float, int, float]]
            ] = []
            for index in range(self.get_number_sites_wo_spin_degree()):
                inner_list: Set[
                    Tuple[int, float, int, float, int, float, int, float]
                ] = set()

                index_circle = set([index])
                for one_step_nb in self.get_nearest_neighbor_indices(index):
                    index_circle.add(one_step_nb)
                    for two_step_nb in self.get_nearest_neighbor_indices(one_step_nb):
                        index_circle.add(two_step_nb)

                for l in index_circle:
                    for m in self.get_nearest_neighbor_indices(l):
                        ab_indices: Set[Tuple[int, int]] = set()

                        # a = l, b is nb of l
                        for b in self.get_nearest_neighbor_indices(l):
                            ab_indices.add((l, b))
                        # a = m, b is nb of m
                        for b in self.get_nearest_neighbor_indices(m):
                            ab_indices.add((m, b))
                        # b = l, a is nb of l & neq m
                        for a in self.get_nearest_neighbor_indices(l):
                            if a != m:
                                ab_indices.add((a, l))
                        # b = m, a is nb of m & neq l
                        for a in self.get_nearest_neighbor_indices(m):
                            if a != l:
                                ab_indices.add((a, m))

                        for a, b in ab_indices:
                            if (
                                l == index or m == index or a == index or b == index
                            ) and (l == a or m == a or l == b or m == b):
                                inner_list.add(
                                    (
                                        l,
                                        self.get_eps_multiplier(
                                            l, phi, sin_phi, cos_phi
                                        ),
                                        m,
                                        self.get_eps_multiplier(
                                            m, phi, sin_phi, cos_phi
                                        ),
                                        a,
                                        self.get_eps_multiplier(
                                            a, phi, sin_phi, cos_phi
                                        ),
                                        b,
                                        self.get_eps_multiplier(
                                            b, phi, sin_phi, cos_phi
                                        ),
                                    )
                                )

                self.index_knows_cache_contains_one.append(inner_list)

        # init index knows tuples that contain at least one of two indices
        if self.index_knows_cache_contains_two is None:
            self.index_knows_cache_contains_two: List[
                List[Set[Tuple[int, float, int, float, int, float, int, float]]]
            ] = []
            for indexa in range(self.get_number_sites_wo_spin_degree()):
                index_list: List[
                    Set[Tuple[int, float, int, float, int, float, int, float]]
                ] = []
                for indexb in range(self.get_number_sites_wo_spin_degree()):
                    inner_list: Set[
                        Tuple[int, float, int, float, int, float, int, float]
                    ] = set()

                    index_circle = set([indexa, indexb])
                    for one_step_nb in self.get_nearest_neighbor_indices(indexa):
                        index_circle.add(one_step_nb)
                        for two_step_nb in self.get_nearest_neighbor_indices(
                            one_step_nb
                        ):
                            index_circle.add(two_step_nb)
                    for one_step_nb in self.get_nearest_neighbor_indices(indexb):
                        index_circle.add(one_step_nb)
                        for two_step_nb in self.get_nearest_neighbor_indices(
                            one_step_nb
                        ):
                            index_circle.add(two_step_nb)

                    for l in index_circle:
                        for m in self.get_nearest_neighbor_indices(l):
                            ab_indices: Set[Tuple[int, int]] = set()

                            # a = l, b is nb of l
                            for b in self.get_nearest_neighbor_indices(l):
                                ab_indices.add((l, b))
                            # a = m, b is nb of m
                            for b in self.get_nearest_neighbor_indices(m):
                                ab_indices.add((m, b))
                            # b = l, a is nb of l & neq m
                            for a in self.get_nearest_neighbor_indices(l):
                                if a != m:
                                    ab_indices.add((a, l))
                            # b = m, a is nb of m & neq l
                            for a in self.get_nearest_neighbor_indices(m):
                                if a != l:
                                    ab_indices.add((a, m))

                            for a, b in ab_indices:
                                if (
                                    l == indexa
                                    or m == indexa
                                    or a == indexa
                                    or b == indexa
                                    or l == indexb
                                    or m == indexb
                                    or a == indexb
                                    or b == indexb
                                ) and (l == a or m == a or l == b or m == b):
                                    inner_list.add(
                                        (
                                            l,
                                            self.get_eps_multiplier(
                                                l, phi, sin_phi, cos_phi
                                            ),
                                            m,
                                            self.get_eps_multiplier(
                                                m, phi, sin_phi, cos_phi
                                            ),
                                            a,
                                            self.get_eps_multiplier(
                                                a, phi, sin_phi, cos_phi
                                            ),
                                            b,
                                            self.get_eps_multiplier(
                                                b, phi, sin_phi, cos_phi
                                            ),
                                        )
                                    )
                    index_list.append(inner_list)
                self.index_knows_cache_contains_two.append(index_list)

        time_end = measure() * 1000
        print(f"Index-Knows-Cache has been pre-computed in {time_end-time_start}ms")

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
        # This will cause an IndexOutOfBoundsException, if there are indices requested, that are not 0<=index<domain_size
        return self.index_knows_cache[index]

    def get_index_knows_tuples_contains_one(
        self, index: int
    ) -> List[Tuple[int, float, int, float, int, float, int, float]]:
        if self.index_knows_cache_contains_one is None:
            raise Exception(
                "Index knows cache is requested, but not yet initialized for this geometry"
            )
        return self.index_knows_cache_contains_one[index]

    def get_index_knows_tuples_contains_two(
        self, indexa: int, indexb: int
    ) -> List[Tuple[int, float, int, float, int, float, int, float]]:
        if self.index_knows_cache_contains_two is None:
            raise Exception(
                "Index knows cache is requested, but not yet initialized for this geometry"
            )
        return self.index_knows_cache_contains_two[indexa][indexb]

    def get_index_overlap_circle_tuples(
        self,
    ) -> Set[Tuple[int, int, int, int]]:
        if self.index_overlap_circle_cache is None:
            raise Exception(
                "Index overlap circle cache is requested, but not yet initialized for this geometry"
            )
        return self.index_overlap_circle_cache


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
        self.size_square = size * size
        super().__init__()

    def get_number_sites_wo_spin_degree(self) -> int:
        return self.size_square

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
        # it makes sense, requesting this for both spin-directions and ensuring correctness by mod
        cut_index = index % self.get_number_sites_wo_spin_degree()
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

        ## TODO: cache
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

        ## TODO: cache
        # it makes sense, requesting this for both spin-directions and ensuring correctness by mod

        domain_size = self.get_number_sites_wo_spin_degree()

        # Attempt to check the lateral symmetry of flipping, by placing the 0-index in the "middle" to make it E-symmetric
        # this should not influence other properties (I think)
        return cos_phi * ((index % domain_size) - (domain_size / 2.0) + 0.5)

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {"type": "LinearChainNonPeriodicState", "size": self.size}
