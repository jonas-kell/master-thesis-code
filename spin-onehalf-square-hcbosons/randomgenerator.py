import random
from typing import Dict


class RandomGenerator:
    def __init__(
        self,
        seed: str,
    ):
        self.seed = seed
        self.random = random.Random(self.seed)

    def randint(self, a: int, b: int):
        """
        Return a random integer N such that a <= N <= b.
        """
        return self.random.randint(a, b)

    def probability(self) -> float:
        """
        Return a random float probability
        """
        return self.random.random()

    def randbool(self) -> bool:
        """
        Return a random boolean
        """
        return self.randint(0, 1) == 1

    def derive(self) -> "RandomGenerator":
        """
        Returns an independent Generator, derived from the current generators seed
        """
        new_seed = (
            f"{self.randint(0, 1000)}{self.randint(0, 1000)}{self.randint(0, 1000)}"
        )
        return RandomGenerator(new_seed)

    def get_log_info(
        self,
    ) -> Dict[str, str]:
        return {
            "seed": self.seed,
        }
