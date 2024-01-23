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


class DoubleOccupation(Observable):
    def __init__(
        self,
    ):
        super().__init__()

    def get_expectation_value(self, system_state_object: state.SystemState) -> float:
        nr_sites = system_state_object.get_number_sites_wo_spin_degree()
        system_state_array = system_state_object.get_state_array()

        running_sum = 0
        for i in range(nr_sites):
            i_os = system_state_object.get_opposite_spin_index(i)

            # only because occupation is either 1 or 0
            running_sum += system_state_array[i] * system_state_array[i_os]

        return running_sum
