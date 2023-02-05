from dataclasses import dataclass
from typing import Optional

import cv2
import nibabel
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
    Class used for storing the metadata of input video/image to restore it for output videos
    """
    width: int
    height: int
    fps: float
    frame_count: int
    fourcc: Optional[str] = None


def load_video(
        video_path: str
):
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
    video_data = np.empty((metadata.frame_count, metadata.height, metadata.width, 3), dtype=float)
    for frame_ind in range(metadata.frame_count):
        ret, frame = cap.read()
        if not ret:
            video_data = video_data[:frame_ind]
            break
        video_data[frame_ind] = frame
    return video_data, metadata


def load_3d_image(
        image_path: str
):
    image = nibabel.load(image_path)
    image = image.get_fdata()
    metadata = VideoMetadata(
        image.shape[2],
        image.shape[1],
        25,
        image.shape[0]
    )
    # Grayscale to RGB
    if len(image.shape) == 3:
        image = np.repeat(image[..., None], 3, -1)
    if image.dtype != np.uint8:
        image = image/image.max() * 255.
        image = image.astype(np.uint8)
    return image, metadata


def parse_input(
        input_path: str,
        n_pieces_x: int,
        n_pieces_y: int,
        n_pieces_z: int,
        input_type: str,
        strict_frame_number: bool = False
):
    """
    Function parsing input data and creating the IndexToDataMapping instance.

    Parameters
    ----------
    input_path : str
        Path to the input data
    n_pieces_x, n_pieces_y, n_pieces_z
        Number of pieces in each dimension. If x
        Size of the puzzle piece in each dimension (width, height being the x, y dimension for each frame, depth being
        number of frames belonging to the piece)
    strict_frame_number : bool
        Default (False) strategy for when the piece_depth would not divide evenly is to drop the additional ending
        frames. When True, if it's impossible to evenly divide the video into pieces with piece_depth dimension, and
        exception is raised.
    input_type : str
        One of "video" or "image" depending on if input is video or 3D image.

    Returns
    -------
    IndexToDataMapping
        Mapping later used to create a population of puzzle solutions
    VideoMetadata
        Metadata of the input video
    """
    if input_type == "video":
        data, metadata = load_video(input_path)
    elif input_type == "image":
        data, metadata = load_3d_image(input_path)
    else:
        raise RuntimeError("Unrecognized input type.")
    piece_width = math.ceil(metadata.width / n_pieces_x)
    new_width = n_pieces_x * piece_width
    piece_height = math.ceil(metadata.height / n_pieces_y)
    new_height = n_pieces_y * piece_height
    piece_depth = math.floor(metadata.frame_count / n_pieces_z)
    if n_pieces_z * piece_depth != metadata.frame_count:
        if strict_frame_number:
            raise RuntimeError(f"Can't divide video with {metadata.frame_count} frames into pieces with depth {piece_depth}.")
        print(f"Can't divide video with {metadata.frame_count} frames into pieces with depth {piece_depth}. "
              f"Dropping {metadata.frame_count - n_pieces_z * piece_depth} frames")
        data = data[:n_pieces_z * piece_depth]

    index_to_data = IndexToDataMapping(
        (n_pieces_x, n_pieces_y, n_pieces_z),
        (piece_width, piece_height, piece_depth),
        input_path
    )
    for frame_ind in range(len(data)):
        frame = cv2.resize(data[frame_ind], (new_width, new_height), interpolation=cv2.INTER_CUBIC)
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
