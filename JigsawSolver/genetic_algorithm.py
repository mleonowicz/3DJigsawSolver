import logging

import numpy as np
from tqdm import trange

from JigsawSolver.core import IndexToDataMapping, Puzzle


class GeneticAlgorithm(object):
    def __init__(
            self,
            mapping: IndexToDataMapping,
            population_size: int,
            logging_level=logging.INFO):
        self.population_size = population_size
        self.mapping = mapping

        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def fit(self, max_iter: int):
        self.logger.info('Started fit function')

        # Create random population
        current_population = [
            Puzzle(self.mapping) for _ in range(self.population_size)
        ]

        # Evaluate population
        current_population_fitness_values = np.array([
            puzzle.fitness for puzzle in current_population
        ])

        best_fitness = float('inf')
        best_puzzle = None
        for t in trange(max_iter):
            self.logger.info(f'Iteration: {t}. Best fitness: {best_fitness}')

            # Choose parents by the roulette method
            fitness_values = current_population_fitness_values - current_population_fitness_values.min()
            if fitness_values.sum() > 0:
                probabilities = fitness_values / fitness_values.sum()
            else:
                probabilities = 1.0 / self.population_size * np.ones(self.population_size)
            parent_indices = np.random.choice(
                self.population_size,
                self.population_size * 2,
                replace=True,
                p=probabilities
            ).reshape((self.population_size, 2))

            new_population = []
            for indices in parent_indices:
                first_parent = indices[0]
                second_parent = indices[1]
                new_puzzle = Puzzle.cross(first_parent, second_parent)
                new_population.appendn(new_puzzle)

            new_population_fitness_values = np.array([
                puzzle.fitness for puzzle in new_population
            ])

            if new_population_fitness_values.min() < best_fitness:
                best_fitness = new_population_fitness_values.min()
                best_puzzle = new_population[new_population_fitness_values.argmin()]  # noqa: F841
                self.logger.info(f'Found new best fitness value: {best_fitness}')

            current_population = new_population
            current_population_fitness_values = new_population_fitness_values
