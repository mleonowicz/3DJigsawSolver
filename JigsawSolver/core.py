import numpy as np


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
