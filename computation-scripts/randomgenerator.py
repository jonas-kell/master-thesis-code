from typing import Dict, List
import random


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

    def normal(self, sigma: float = 1) -> float:
        """
        Return a float, pulled from a binomial distribution
        """
        return self.random.normalvariate(mu=0, sigma=sigma)

    def randbool(self) -> bool:
        """
        Return a random boolean
        """
        return self.randint(0, 1) == 1

    def rand_occupation_array(self, length: int) -> List[int]:
        """
        Return an array of 0 or 1, each randomly chosen with equal probability
        """
        return self.random.choices([0, 1], weights=[0.5, 0.5], k=length)

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

    def binomial_random(self, n: int):
        successes = 0
        for _ in range(n):
            if self.probability() < 0.5:
                successes += 1
        return successes
