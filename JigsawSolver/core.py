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


def calculate_dissimilarity(piece_a: PuzzlePiece, piece_b: PuzzlePiece, orientation: str) -> int:
    """
    Calculates dissimilarity betweeen two puzzles given an orientation.

    Parameters
    ----------
    piece_a : PuzzlePiece
        First puzzle to compare
    piece_b : PuzzlePiece
        Second puzzle to compare
    orientation : str
        Orientation of the puzzles. There are three valid arguments:
        * 'LR' = Left - Right
        * 'UP' = Up - Down
        * 'BF' = Back - Forward
    """

    assert orientation in ['LR', 'UP', 'BF']
    piece_a = piece_a.piece
    piece_b = piece_b.piece

    height, width, depth, _ = piece_a.shape
    if orientation == 'LR':
        color_difference = piece_a[:, width - 1, :, :] - piece_b[:, 0, :, :]
    if orientation == 'UP':
        color_difference = piece_a[height - 1, :, :, :] - piece_b[0, :, :, :]
    if orientation == 'BF':
        color_difference = piece_a[:, :, depth - 1, :] - piece_b[:, :, 0, :]

    color_difference = np.power(color_difference, 2)
    total_difference = np.sum(color_difference)
    dissimilarity = np.sqrt(total_difference)

    return dissimilarity
