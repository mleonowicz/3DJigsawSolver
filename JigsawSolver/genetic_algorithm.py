import logging

import numpy as np
from tqdm import trange

from JigsawSolver.core import IndexToDataMapping, Puzzle, CrossOperator


class GeneticAlgorithm(object):
    def __init__(
            self,
            mapping: IndexToDataMapping,
            population_size: int,
            elites: int = None,
            logging_level=logging.INFO):
        self.population_size = population_size
        if not elites:
            self.elites = self.population_size // 2
        else:
            self.elites = elites
        self.mapping = mapping

        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def fit(self, max_iter: int, max_no_change_iter: int = 5):
        self.logger.info('Started fit function')

        original_puzzle = Puzzle(
            self.mapping,
            np.arange(
                self.mapping.num_of_puzzles, dtype=np.int32
            ).reshape(self.mapping.n_pieces_x, self.mapping.n_pieces_y, self.mapping.n_pieces_z)
        )
        original_fitness = original_puzzle.fitness
        self.logger.info(f'Optimal fitness: {original_fitness}')

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
        no_change_counter = 0
        for t in trange(max_iter):
            self.logger.info(f'Iteration: {t}. Best fitness: {best_fitness}')

            # Choosing elites
            elites = list(sorted(current_population, key=lambda p: p.fitness)[:self.elites])
            new_population = elites

            # Choose parents by the roulette method
            fitness_values = current_population_fitness_values - current_population_fitness_values.min()
            if fitness_values.sum() > 0:
                probabilities = current_population_fitness_values / current_population_fitness_values.sum()
            else:
                probabilities = 1.0 / self.population_size * np.ones(self.population_size)
            parent_indices = np.random.choice(
                self.population_size,
                (self.population_size - len(new_population)) * 2,
                replace=True,
                p=probabilities
            ).reshape((-1, 2))

            for indices in parent_indices:
                first_parent = current_population[indices[0]]
                second_parent = current_population[indices[1]]
                cross_operator = CrossOperator(first_parent, second_parent)
                new_puzzle = cross_operator()
                new_population.append(new_puzzle)

            new_population_fitness_values = np.array([
                puzzle.fitness for puzzle in new_population
            ])

            if new_population_fitness_values.min() < best_fitness:
                best_fitness = new_population_fitness_values.min()
                best_puzzle = new_population[new_population_fitness_values.argmin()]  # noqa: F841
                self.logger.info(f'Found new best fitness value: {best_fitness}')
                no_change_counter = 0
            else:
                no_change_counter += 1

            if no_change_counter >= max_no_change_iter:
                self.logger.info(f'No change for {no_change_counter} iterations. Ending prematurely.')
                return best_fitness, best_puzzle

            current_population = new_population
            current_population_fitness_values = new_population_fitness_values
        return best_fitness, best_puzzle
