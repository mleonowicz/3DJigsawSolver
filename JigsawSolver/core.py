from itertools import product
from typing import Tuple

import numpy as np

from JigsawSolver.dissimilarity import calculate_dissimilarity


class IndexToDataMapping:
    def __init__(self,
                 n_pieces: Tuple[int, int, int],
                 piece_size: Tuple[int, int, int]):
        """
        Mapping between the index of the piece and the video data.

        Parameters
        ----------
        n_pieces : Tuple[int, int, int]
            Number of pieces in a puzzle in each dimension
        piece_size : Tuple[int, int, int]
            Sizes of the pieces in each dimension
        """
        self.n_pieces_x, self.n_pieces_y, self.n_pieces_z = n_pieces
        self.num_of_puzzles = self.n_pieces_x * self.n_pieces_y * self.n_pieces_z
        self.width, self.height, self.depth = piece_size
        self.id_map = {}
        self.dissimilarity_cache = {}
        for index in range(self.num_of_puzzles):
            self.id_map[index] = np.empty((self.height, self.width, self.depth, 3), dtype=np.uint8)

    def coords_to_index(self, coords: Tuple[int, int, int]) -> int:
        """
        Mapping that converts given coordinates into a index of a puzzle.

        Parameters
        ----------
        coords : Tuple[int, int, int]
            (x, y, z) coordinates of the puzzle in the original image

        Returns
        -------
        int
            Index of the puzzle in the `id_map`
        """
        xcoord, ycoord, zcoord = coords
        return xcoord * self.n_pieces_y * self.n_pieces_z + ycoord * self.n_pieces_z + zcoord

    def index_to_coords(self, index: int) -> Tuple[int, int, int]:
        """
        Mapping that converts given index into coordinates of a puzzle.

        Parameters
        ----------
        index : int
            Index of the puzzle in the `id_map`

        Returns
        -------
        Tuple[int, int, int]
            Coordinates of the puzzle of id `index` in the original image
        """
        zcoord = index % self.n_pieces_z
        ycoord = (index // self.n_pieces_z) % self.n_pieces_y
        xcoord = (index // (self.n_pieces_z * self.n_pieces_y)) % self.n_pieces_x
        return (xcoord, ycoord, zcoord)

    def add_frame(self, frame: np.ndarray, frame_count):
        """
        Method used while building the mapping, it should be called for each frame in the video in order
        to correctly fill the data for all the puzzle pieces.

        Parameters
        ----------
        frame : np.ndarray
            Frame of the analyzed video
        frame_count : int
            Index of the frame in the video
        """
        zcoord, data_z_ind = frame_count // self.depth, frame_count % self.depth
        for xcoord, ycoord in product(range(self.n_pieces_x), range(self.n_pieces_y)):
            index = self.coords_to_index((xcoord, ycoord, zcoord))
            puzzle_data = self.id_map[index]
            puzzle_data[:, :, data_z_ind] = frame[self.height * ycoord: self.height * (ycoord + 1),
                                                  self.width * xcoord: self.width * (xcoord + 1)]

    def __getitem__(self, index):
        return self.id_map[index]

    def get_dissimilarity(self, index_a: int, index_b: int, orientation: str) -> float:
        """
        Calculates dissimilarity betweeen puzzles of `index_a` and `index_b` ids given an orientation.
        This function caches the calculated values.

        Parameters
        ----------
        index_a : int
            Index of the first puzzle to compare.
        piece_b : int
            Index of the second puzzle to compare.
        orientation : str
            Orientation of the puzzles. There are three valid arguments:
            * 'LR' = Left - Right
            * 'UD' = Up - Down
            * 'FB' = Forward - Back

        Returns
        -------
        float
            Calculated dissimilarity
        """
        key = (index_a, index_b, orientation)
        try:
            return self.dissimilarity_cache[key]
        except KeyError:
            dissimilarity = calculate_dissimilarity(self[index_a], self[index_b], orientation)
            self.dissimilarity_cache[key] = dissimilarity
            return dissimilarity


class Puzzle:
    def __init__(self, mapping: IndexToDataMapping, puzzle_pieces: np.ndarray = None):
        """
        Representation of a single solution to the puzzle

        Parameters
        ----------
        mapping : IndexToDataMapping
            Mapping between the puzzle indexes and the video data corresponding to each piece
        puzzle_pieces : np.ndarray
            3D numpy array that with indices of puzzles
        """
        self._fitness = None
        self.index_to_data = mapping
        self.n_x, self.n_y, self.n_z = mapping.n_pieces_x, mapping.n_pieces_y, mapping.n_pieces_z
        if puzzle_pieces is not None:
            self.puzzle = puzzle_pieces
        else:
            # shuffling the pieces
            self.puzzle = np.arange(mapping.num_of_puzzles, dtype=np.int32)
            np.random.shuffle(self.puzzle)
            self.puzzle = np.reshape(self.puzzle, (self.n_x, self.n_y, self.n_z))

    @property
    def fitness(self) -> float:
        """
        Fitness value getter that is calculated by summing dissimilarity between all adjecent pieces.

        Returns
        -------
        float
            Fitness value
        """
        if self._fitness:
            return self._fitness

        self._fitness = 0
        for i in range(self.n_x - 1):
            for j in range(self.n_y):
                for k in range(self.n_z):
                    index_a = self.puzzle[i][j][k]
                    index_b = self.puzzle[i + 1][j][k]
                    self._fitness += self.index_to_data.get_dissimilarity(index_a, index_b, 'LR')

        for i in range(self.n_x):
            for j in range(self.n_y - 1):
                for k in range(self.n_z):
                    index_a = self.puzzle[i][j][k]
                    index_b = self.puzzle[i][j + 1][k]
                    self._fitness += self.index_to_data.get_dissimilarity(index_a, index_b, 'UD')

        for i in range(self.n_x):
            for j in range(self.n_y):
                for k in range(self.n_z - 1):
                    index_a = self.puzzle[i][j][k]
                    index_b = self.puzzle[i][j][k + 1]
                    self._fitness += self.index_to_data.get_dissimilarity(index_a, index_b, 'FB')

        return self._fitness

    @classmethod
    def cross(cls, first_parent, second_parent):
        pass
