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
        First puzzle to compare.
    piece_b : PuzzlePiece
        Second puzzle to compare.
    orientation : str
        Orientation of the puzzles. There are three valid arguments:
        * 'LR' = Left - Right
        * 'UD' = Up - Down
        * 'BF' = Back - Forward
    """

    assert orientation in ['LR', 'UD', 'BF']
    piece_a = piece_a.piece.astype('int64')
    piece_b = piece_b.piece.astype('int64')

    height, width, depth, _ = piece_a.shape
    if orientation == 'LR':
        color_difference = (piece_a[:, width - 1, :, :] - piece_b[:, 0, :, :]).astype('int64')
    if orientation == 'UD':
        color_difference = (piece_a[height - 1, :, :, :] - piece_b[0, :, :, :]).astype('int64')
    if orientation == 'BF':
        color_difference = (piece_a[:, :, depth - 1, :] - piece_b[:, :, 0, :]).astype('int64')

    color_difference = np.power(color_difference, 2)
    total_difference = np.sum(color_difference)
    dissimilarity = np.sqrt(total_difference)

    return dissimilarity


frame_a = np.asarray(
    [
        [
            [1, 1, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [1, 1, 1]
        ]
    ]
)

frame_b = frame_a.copy()
frame_b[0][0] = [0, 0, 0]
frame_b[0][1] = [0, 0, 0]
frame_b[1][1] = [0, 0, 0]

# frame_a 3x2x1
#
# 1,1,1   1,1,1
# 1,1,1   1,1,1
# 1,1,1   1,1,1

# frame_b 3x2x1
#
# 0,0,0   0,0,0
# 1,1,1   0,0,0
# 1,1,1   1,1,1

piece_a = PuzzlePiece((0, 0, 0), (2, 3, 1))
piece_a.add_frame(frame_a)

piece_b = PuzzlePiece((0, 0, 0), (2, 3, 1))
piece_b.add_frame(frame_b)

assert calculate_dissimilarity(piece_a, piece_b, 'LR') == np.sqrt(3)
assert calculate_dissimilarity(piece_b, piece_a, 'LR') == np.sqrt(6)

assert calculate_dissimilarity(piece_a, piece_b, 'UD') == np.sqrt(6)
assert calculate_dissimilarity(piece_b, piece_a, 'UD') == np.sqrt(0)

assert calculate_dissimilarity(piece_b, piece_a, 'BF') == np.sqrt(9)
assert calculate_dissimilarity(piece_b, piece_a, 'BF') == np.sqrt(9)
