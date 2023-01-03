from itertools import product
from typing import Tuple

import numpy as np


class IndexToDataMapping:
    def __init__(self,
                 n_pieces: Tuple[int, int, int],
                 piece_size: Tuple[int, int, int]):
        """
        Mapping between the index of the piece and the video data. Key is the triple of integers that indicate the
        position of the puzzle piece in the original video, the value is the numpy array of the [x, y, z, 3] dimension,
        [x, y] being the slice of each frame, the z dimension being the frame that belong to the puzzle piece.

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


class Puzzle:
    def __init__(self, mapping: IndexToDataMapping, puzzle_pieces=None):
        """
        Representation of a single solution to the puzzle

        Parameters
        ----------
        mapping : IndexToDataMapping
            Mapping between the puzzle indexes and the video data corresponding to each piece
        puzzle_pieces
            3D numpy array that with indices of puzzles
        """
        self.index_to_data = mapping
        self.n_x, self.n_y, self.n_z = mapping.n_pieces_x, mapping.n_pieces_y, mapping.n_pieces_z
        if puzzle_pieces is not None:
            self.puzzle = puzzle_pieces
        else:
            # shuffling the pieces
            self.puzzle = np.arange(mapping.num_of_puzzles, dtype=np.int32)
            np.random.shuffle(self.puzzle)
            self.puzzle = np.reshape(self.puzzle, (self.n_x, self.n_y, self.n_z))

    def fitness(self):
        pass

    @classmethod
    def cross(cls, parent1, parent2):
        pass
