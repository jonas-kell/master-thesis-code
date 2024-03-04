from abc import ABC, abstractmethod
import state


class Observable(ABC):
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def get_expectation_value(self, system_state_object: state.SystemState) -> float:
        pass


class DoubleOccupationFraction(Observable):
    def __init__(
        self,
    ):
        super().__init__()

    def get_expectation_value(self, system_state_object: state.SystemState) -> float:
        nr_sites = system_state_object.get_number_sites_wo_spin_degree()
        system_state_array = system_state_object.get_state_array()
        domain_size = system_state_object.get_number_sites_wo_spin_degree()

        running_sum = 0
        for i in range(nr_sites):
            i_os = system_state_object.get_opposite_spin_index(i)

            # only because occupation is either 1 or 0
            running_sum += system_state_array[i] * system_state_array[i_os]

        return running_sum / domain_size


class DoubleOccupationAtSite(Observable):

    def __init__(self, site: int, system_state_object: state.SystemState):
        super().__init__()

        if site < 0:
            raise Exception("Site must be at least 0")

        domain_size = system_state_object.get_number_sites_wo_spin_degree()
        if site >= domain_size:
            raise Exception(f"Site must be smaller than {domain_size} to fit")

        self.site = site

    def get_expectation_value(self, system_state_object: state.SystemState) -> float:
        system_state_array = system_state_object.get_state_array()

        return (
            system_state_array[self.site]
            * system_state_array[system_state_object.get_opposite_spin_index(self.site)]
        )
