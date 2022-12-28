import logging

from tqdm import trange


class GeneticAlgorithm(object):
    def __init__(self, logging_level=logging.INFO):
        logging.basicConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def fit(self, max_iter: int):
        self.logger.info('Started fit function')

        best_fitness = float('-inf')
        for t in trange(max_iter):
            self.logger.info(f'Iteration: {t}. Best fitness: {best_fitness}')
