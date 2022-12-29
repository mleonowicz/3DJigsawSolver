import itertools

import cv2

from JigsawSolver.core import PuzzlePiece


def create_puzzle(video_path, piece_width, piece_height, piece_length):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open a {video_path} video")
    puzzle = []
    n_pieces_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / piece_width)
    n_pieces_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / piece_height)
    # TODO: Warning when frame_width/height/video_length % piece_width/height/length != 0
    temp_pieces = [
        PuzzlePiece((xcoord, ycoord, 0), (piece_width, piece_height, piece_length))
        for xcoord, ycoord in itertools.product(range(n_pieces_x), range(n_pieces_y))
    ]
    for frame_ind in itertools.count(1):
        ret, frame = cap.read()
        if not ret:
            break
        for temp_piece in temp_pieces:
            temp_piece.add_frame(frame)
        if frame_ind % piece_length == 0:
            puzzle.extend(temp_pieces)
            temp_pieces = [
                PuzzlePiece((xcoord, ycoord, frame_ind // piece_length), (piece_width, piece_height, piece_length))
                for xcoord, ycoord in itertools.product(range(n_pieces_x), range(n_pieces_y))
            ]
    return puzzle
