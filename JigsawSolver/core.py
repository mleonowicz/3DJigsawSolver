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
    def __init__(self, coords, piece_size):
        self.x_coord, self.y_coord, self.z_coord = coords
        self.width, self.height, self.length = piece_size
        self.piece = np.empty((self.height, self.width, self.length, 3), dtype=np.uint8)
        self._frame_ind = 0

    def add_frame(self, frame):
        self.piece[:, :, self._frame_ind] = frame[self.height * self.y_coord: self.height * (self.y_coord + 1),
                                                  self.width * self.x_coord: self.width * (self.x_coord + 1)]
        self._frame_ind += 1
