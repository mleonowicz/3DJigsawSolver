import logging
import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import trange

from JigsawSolver.core import IndexToDataMapping, Puzzle, CrossOperator


class GeneticAlgorithm(object):
    def __init__(
            self,
            mapping: IndexToDataMapping,
            population_size: int,
            elites: int = None,
            alpha: float = 0.005,
            beta: float = 0.05,
            logging_level=logging.INFO,
            output_path='output.dat'):
        self.population_size = population_size
        if not elites:
            self.elites = self.population_size // 2
        else:
            self.elites = elites
        self.mapping = mapping
        self.alpha = alpha
        self.beta = beta
        self.output_path = f'{output_path}_{time.time()}'

        self.min_fitness_history = []
        self.mean_fitness_history = []
        self.max_fitness_history = []

        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def save_history(self):
        with open(f'{self.output_path}.min', 'w+') as f:
            f.write(', '.join(map(str, self.min_fitness_history)))

        with open(f'{self.output_path}.mean', 'w+') as f:
            f.write(', '.join(map(str, self.mean_fitness_history)))

        with open(f'{self.output_path}.max', 'w+') as f:
            f.write(', '.join(map(str, self.max_fitness_history)))

    def draw_history(self):
        xs = list(range(0, len(self.min_fitness_history)))
        plt.plot(xs, self.min_fitness_history, marker='o', label='Min fitness')
        plt.plot(xs, self.mean_fitness_history, marker='o', label='Mean fitness')
        plt.plot(xs, self.max_fitness_history, marker='o', label='Max fitness')
        plt.title('Fitness values')
        plt.ylabel('Fitness')
        plt.xlabel('Iteration')
        plt.legend(loc="upper right")
        plt.savefig(f'{self.output_path}.history.png', bbox_inches='tight')

    def fit(self, max_iter: int, max_no_change_iter: int = 20):
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

        self.min_fitness_history = [current_population_fitness_values.min()]
        self.mean_fitness_history = [current_population_fitness_values.mean()]
        self.max_fitness_history = [current_population_fitness_values.max()]
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

            # Creating new offsprings
            new_offsprings = []
            for indices in parent_indices:
                first_parent, second_parent = current_population[indices[0]], current_population[indices[1]]
                cross_operator = CrossOperator(first_parent, second_parent, self.alpha)
                new_puzzle = cross_operator()
                if np.random.uniform() < self.beta and self.mapping.n_pieces_z > 1:
                    left, right = np.random.randint(low=0, high=self.mapping.n_pieces_z + 1, size=2)
                    if left > right:
                        left, right = right, left
                    new_puzzle.puzzle[:, :, left:right] = new_puzzle.puzzle[:, :, left:right][:, :, ::-1]
                new_offsprings.append(new_puzzle)

            new_population = new_population + new_offsprings

            new_population_fitness_values = np.array([
                puzzle.fitness for puzzle in new_population
            ])

            self.min_fitness_history.append(new_population_fitness_values.min())
            self.mean_fitness_history.append(new_population_fitness_values.mean())
            self.max_fitness_history.append(new_population_fitness_values.max())

            if new_population_fitness_values.min() < best_fitness:
                best_fitness = new_population_fitness_values.min()
                best_puzzle = new_population[new_population_fitness_values.argmin()]  # noqa: F841
                self.logger.info(f'Found new best fitness value: {best_fitness}')
                no_change_counter = 0
            else:
                no_change_counter += 1

            if no_change_counter >= max_no_change_iter:
                self.logger.info(f'No change for {no_change_counter} iterations. Ending prematurely.')
                self.save_history()
                self.draw_history()
                return best_fitness, best_puzzle

            current_population = new_population
            current_population_fitness_values = new_population_fitness_values

        self.save_history()
        self.draw_history()
        return best_fitness, best_puzzle
