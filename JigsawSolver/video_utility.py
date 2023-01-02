import itertools

import cv2

from JigsawSolver.core import IndexToDataMapping


def parse_video(video_path: str, piece_width: int, piece_height: int, piece_depth: int):
    """
    Function parsing input video and creating the IndexToDataMapping instance.

    Parameters
    ----------
    video_path : str
        Path to the input video
    piece_width, piece_height, piece_depth
        Size of the puzzle piece in each dimension (width, height being the x, y dimension for each frame, depth being
        number of frames belonging to the piece)

    Returns
    -------
    IndexToDataMapping
        Mapping later used to create a population of puzzle solutions.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open a {video_path} video")
    n_pieces_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / piece_width)
    n_pieces_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / piece_height)
    n_pieces_z = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / piece_depth)
    # TODO: Warning when frame_width/height/video_depth % piece_width/height/depth != 0
    index_to_data = IndexToDataMapping((n_pieces_x, n_pieces_y, n_pieces_z), (piece_width, piece_height, piece_depth))
    for frame_ind in itertools.count(1):
        ret, frame = cap.read()
        if not ret:
            break
        index_to_data.add_frame(frame, frame_ind)
    return index_to_data
