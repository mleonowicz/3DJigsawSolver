import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from JigsawSolver.core import IndexToDataMapping, Puzzle


def _get_fourcc(cap):
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # translating from byte representation to 4-character string
    fourcc = bytes([
        (fourcc >> 8*shift) & 255 for shift in range(4)
    ]).decode()
    return fourcc


@dataclass
class VideoMetadata:
    """
    Class used for storing the metadata of input video to restore it for output videos
    """
    width: int
    height: int
    fps: float
    frame_count: int
    fourcc: str


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
        Mapping later used to create a population of puzzle solutions
    VideoMetadata
        Metadata of the input video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open a {video_path} video")
    metadata = VideoMetadata(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        _get_fourcc(cap)
    )
    n_pieces_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / piece_width)
    n_pieces_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / piece_height)
    n_pieces_z = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / piece_depth)
    # TODO: Warning when frame_width/height/video_depth % piece_width/height/depth != 0
    index_to_data = IndexToDataMapping((n_pieces_x, n_pieces_y, n_pieces_z), (piece_width, piece_height, piece_depth))
    for frame_ind in itertools.count():
        ret, frame = cap.read()
        if not ret:
            break
        index_to_data.add_frame(frame, frame_ind)
    return index_to_data, metadata


def save_puzzle_video(
        output_path: str,
        puzzle: Puzzle,
        metadata: VideoMetadata,
        piece_width: int,
        piece_height: int,
        piece_depth: int
):
    """
    Takes the solution stored in puzzle and visualize it through the video

    Parameters
    ----------
    output_path : str
        Path to the output video
    puzzle : Puzzle
        Puzzle solution to visualize
    metadata : Metadata of the input video, result of parse_video
    piece_width, piece_height, piece_depth : int
        Size of the puzzle piece in each dimension
    """
    # fourcc = cv2.VideoWriter_fourcc(*metadata.fourcc)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")  # Tested locally only with .avi files
    writer = cv2.VideoWriter(output_path, fourcc, metadata.fps, (metadata.width, metadata.height))
    if not writer.isOpened():
        raise RuntimeError("Could not save the video")
    # Should test if memory allows for bigger videos
    output_video = np.empty((metadata.height, metadata.width, metadata.frame_count, 3), dtype=np.uint8)

    for coords, index in np.ndenumerate(puzzle.puzzle):
        xcoord, ycoord, zcoord = coords
        output_video[
            piece_height*ycoord: piece_height*(ycoord + 1),
            piece_width*xcoord: piece_width*(xcoord + 1),
            piece_depth*zcoord: piece_depth*(zcoord + 1)
        ] = puzzle.index_to_data[index]
    for frame_ind in range(metadata.frame_count):
        writer.write(output_video[:, :, frame_ind])
    writer.release()
