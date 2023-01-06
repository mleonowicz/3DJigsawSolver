import numpy as np


def calculate_dissimilarity(piece_a: np.ndarray, piece_b: np.ndarray, orientation: str) -> float:
    """
    Calculates dissimilarity betweeen two puzzles given an orientation.

    Parameters
    ----------
    piece_a : np.ndarray
        First puzzle to compare.
    piece_b : np.ndarray
        Second puzzle to compare.
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

    assert orientation in ['LR', 'UD', 'FB']
    piece_a = piece_a.astype('int64')
    piece_b = piece_b.astype('int64')

    height, width, depth, _ = piece_a.shape
    if orientation == 'LR':
        color_difference = (piece_a[:, width - 1, :, :] - piece_b[:, 0, :, :]).astype('int64')
    if orientation == 'UD':
        color_difference = (piece_a[height - 1, :, :, :] - piece_b[0, :, :, :]).astype('int64')
    if orientation == 'FB':
        color_difference = (piece_a[:, :, depth - 1, :] - piece_b[:, :, 0, :]).astype('int64')

    color_difference = np.power(color_difference, 2)
    total_difference = np.sum(color_difference)
    dissimilarity = np.sqrt(total_difference)

    return dissimilarity
