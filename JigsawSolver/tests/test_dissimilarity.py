import numpy as np
import pytest

from JigsawSolver.core import PuzzlePiece
from JigsawSolver.dissimilarity import calculate_dissimilarity


@pytest.fixture
def pieces():
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

    piece_a = PuzzlePiece((0, 0, 0), (2, 3, 1))
    piece_a.add_frame(frame_a)

    piece_b = PuzzlePiece((0, 0, 0), (2, 3, 1))
    piece_b.add_frame(frame_b)
    return piece_a, piece_b


def test_dissimilarity_function(pieces):
    piece_a, piece_b = pieces
    assert calculate_dissimilarity(piece_a, piece_b, 'LR') == np.sqrt(3)
    assert calculate_dissimilarity(piece_b, piece_a, 'LR') == np.sqrt(6)

    assert calculate_dissimilarity(piece_a, piece_b, 'UD') == np.sqrt(6)
    assert calculate_dissimilarity(piece_b, piece_a, 'UD') == np.sqrt(0)

    assert calculate_dissimilarity(piece_b, piece_a, 'BF') == np.sqrt(9)
    assert calculate_dissimilarity(piece_b, piece_a, 'BF') == np.sqrt(9)
