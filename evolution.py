import numpy as np
from typing import Callable

from population import Population
from questionset import Questionset


class Evolution:
    def __init__(
        self,
        answers: np.ndarray,
        fitness_func: Callable,
        n_questions_1: int,
        n_questions_2: int,
        n_switches: int,
        pool_size: int,
        results: np.ndarray,
    ):
        self.n_switches = n_switches

        self.population = Population(
            answers, fitness_func, n_questions_1, n_questions_2, pool_size, results
        )

    def step(self) -> None:
        """
        Do 1 evolution step. This consists of:
        1. Copy and mutate every question set
        2. Add the mutated copies to the population
        3. Sort the population by how good the question set performs
        4. Remove the worst performing individuals
        """
        offsprings = []
        for parent in self.population.individuals:
            offsprings.append(parent.mutate(n_switches=self.n_switches))

        self.population.individuals.extend(offsprings)
        self.population.sort_individuals()
        self.population.remove_individuals()
        return

    def best(self) -> Questionset:
        """Returns the best solution"""
        return self.population.individuals[-1]
