from abc import ABC, abstractmethod
import state
import systemgeometry
from typing import Dict, Union, Any


class Observable(ABC):
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def get_expectation_value(self, system_state: state.SystemState) -> float:
        pass

    @abstractmethod
    def get_label(self) -> str:
        pass

    @abstractmethod
    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        pass


class DoubleOccupationFraction(Observable):
    def __init__(
        self,
    ):
        super().__init__()

    def get_expectation_value(self, system_state: state.SystemState) -> float:
        nr_sites = system_state.get_number_sites_wo_spin_degree()
        system_state_array = system_state.get_state_array()
        domain_size = system_state.get_number_sites_wo_spin_degree()

        running_sum = 0
        for i in range(nr_sites):
            i_os = system_state.get_opposite_spin_index(i)

            # only because occupation is either 1 or 0
            running_sum += system_state_array[i] * system_state_array[i_os]

        return running_sum / domain_size

    def get_label(self) -> str:
        return "Average amount of double Occupation"

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {"type": "DoubleOccupationFraction", "label": self.get_label()}


class DoubleOccupationAtSite(Observable):

    def __init__(self, site: int, system_geometry: systemgeometry.SystemGeometry):
        super().__init__()

        if site < 0:
            raise Exception("Site must be at least 0")

        domain_size = system_geometry.get_number_sites_wo_spin_degree()
        if site >= domain_size:
            raise Exception(f"Site must be smaller than {domain_size} to fit")

        self.site = site

    def get_expectation_value(self, system_state: state.SystemState) -> float:
        system_state_array = system_state.get_state_array()

        return (
            system_state_array[self.site]
            * system_state_array[system_state.get_opposite_spin_index(self.site)]
        )

    def get_label(self) -> str:
        return f"Double Occupation at site {self.site}"

    def get_log_info(self) -> Dict[str, Union[float, str, Dict[Any, Any]]]:
        return {
            "type": "DoubleOccupationAtSite",
            "label": self.get_label(),
            "site": self.site,
        }
