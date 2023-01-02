from itertools import product
from random import shuffle
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
        self.width, self.height, self.depth = piece_size
        self.map = {}
        for xcoord, ycoord, zcoord in product(range(self.n_pieces_x), range(self.n_pieces_y), range(self.n_pieces_z)):
            self.map[(xcoord, ycoord, zcoord)] = np.empty((self.height, self.width, self.depth, 3))

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
            puzzle_data = self.map[(xcoord, ycoord, zcoord)]
            puzzle_data[:, :, data_z_ind] = frame[self.height * ycoord: self.height * (ycoord + 1),
                                                  self.width * xcoord: self.width * (xcoord + 1)]

    def __getitem__(self, key):
        return self.map[key]


class PuzzlePiece:
    def __init__(self, piece_index, position):
        self.index = piece_index
        self.xcoord, self.ycoord, self.zcoord = position


class Puzzle:
    def __init__(self, mapping: IndexToDataMapping, puzzle_pieces=None):
        """
        Representation of a single solution to the puzzle
        
        Parameters
        ----------
        mapping : IndexToDataMapping
            Mapping between the puzzle indexes and the video data corresponding to each piece
        puzzle_pieces
            Order in which puzzles should be arranged in this particular solution. If None, pieces are randomly arranged
        """
        self.index_to_data = mapping
        n_x, n_y, n_z = mapping.n_pieces_x, mapping.n_pieces_y, mapping.n_pieces_z
        if puzzle_pieces is not None:
            # I'm thinking this should be the way of creating the Puzzle instance when crossing two population members
            # Not sure how to represent the singular pieces for crossing to be comfortable, list of PuzzlePiece may be
            # awkward to cross.
            self.puzzle = puzzle_pieces
        else:
            # shuffling the pieces
            position_indexes = list(product(range(n_x), range(n_y), range(n_z)))
            shuffle(position_indexes)
            self.puzzle = []
            for index, position in zip(product(range(n_x), range(n_y), range(n_z)), position_indexes):
                self.puzzle.append(PuzzlePiece(index, position))

    def fitness(self):
        pass

    def mutate(self):
        pass

    @classmethod
    def cross(cls, parent1, parent2):
        pass

    def create_video(self):
        pass
