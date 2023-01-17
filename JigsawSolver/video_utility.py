import itertools
from dataclasses import dataclass

import cv2
import numpy as np
import math

from JigsawSolver.core import IndexToDataMapping, Puzzle


def _get_fourcc(cap):
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # translating from byte representation to 4-character string
    fourcc = bytes([
        (fourcc >> 8 * shift) & 255 for shift in range(4)
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


def parse_video(
        video_path: str,
        n_pieces_x: int,
        n_pieces_y: int,
        n_pieces_z: int,
        strict_frame_number: bool = False
):
    """
    Function parsing input video and creating the IndexToDataMapping instance.

    Parameters
    ----------
    video_path : str
        Path to the input video
    n_pieces_x, n_pieces_y, n_pieces_z
        Number of pieces in each dimension. If x
        Size of the puzzle piece in each dimension (width, height being the x, y dimension for each frame, depth being
        number of frames belonging to the piece)
    strict_frame_number : bool
        Default (False) strategy for when the piece_depth would not divide evenly is to drop the additional ending
        frames. When True, if it's impossible to evenly divide the video into pieces with piece_depth dimension, and
        exception is raised.

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
    piece_width = math.ceil(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / n_pieces_x)
    new_width = n_pieces_x * piece_width
    piece_height = math.ceil(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / n_pieces_y)
    new_height = n_pieces_y * piece_height
    piece_depth = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT) / n_pieces_z)
    if n_pieces_z * piece_depth != metadata.frame_count:
        if strict_frame_number:
            raise RuntimeError(f"Can't divide video with {metadata.frame_count} frames into pieces with depth {piece_depth}.")
        print(f"Can't divide video with {metadata.frame_count} frames into pieces with depth {piece_depth}. "
              f"Dropping {metadata.frame_count - n_pieces_z * piece_depth} frames")

    index_to_data = IndexToDataMapping((n_pieces_x, n_pieces_y, n_pieces_z), (piece_width, piece_height, piece_depth))
    for frame_ind in range(n_pieces_z * piece_depth):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        index_to_data.add_frame(frame, frame_ind)
    return index_to_data, metadata


def save_puzzle_video(
        output_path: str,
        puzzle: Puzzle,
        metadata: VideoMetadata
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
    """
    # fourcc = cv2.VideoWriter_fourcc(*metadata.fourcc)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")  # Tested locally only with .avi files
    writer = cv2.VideoWriter(output_path, fourcc, metadata.fps, (metadata.width, metadata.height))
    if not writer.isOpened():
        raise RuntimeError("Could not save the video")

    n_x, n_y, n_z = puzzle.n_x, puzzle.n_y, puzzle.n_z
    piece_width = math.ceil(metadata.width / n_x)
    piece_height = math.ceil(metadata.height / n_y)
    piece_depth = metadata.frame_count // n_z

    puzzle_width = puzzle.index_to_data.width * puzzle.n_x
    puzzle_height = puzzle.index_to_data.height * puzzle.n_y

    # Should test if memory allows for bigger videos
    output_video = np.empty((puzzle_height, puzzle_width, metadata.frame_count, 3), dtype=np.uint8)

    for coords, index in np.ndenumerate(puzzle.puzzle):
        xcoord, ycoord, zcoord = coords
        output_video[
            piece_height*ycoord: piece_height*(ycoord + 1),
            piece_width*xcoord: piece_width*(xcoord + 1),
            piece_depth*zcoord: piece_depth*(zcoord + 1)
        ] = puzzle.index_to_data[index]
    for frame_ind in range(metadata.frame_count):
        frame = cv2.resize(output_video[:, :, frame_ind], (metadata.width, metadata.height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
