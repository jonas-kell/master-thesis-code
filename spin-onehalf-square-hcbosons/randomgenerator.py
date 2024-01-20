import random


class RandomGenerator:
    def __init__(
        self,
        seed: str,
    ):
        self.seed = seed
        random.seed(a=seed)

    def randint(self, a: int, b: int):
        """
        Return a random integer N such that a <= N <= b.
        """
        return random.randint(a, b)

    def probability(self) -> float:
        """
        Return a random float probability
        """
        return random.random()
