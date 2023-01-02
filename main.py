import argparse

from JigsawSolver.core import Puzzle
from JigsawSolver.video_utility import parse_video, save_puzzle_video


def main():
    parser = argparse.ArgumentParser(
        prog='3DJigsawSolver'
    )
    parser.add_argument(
        '--video-path',
        type=str,
        default='example/example.mp4',
    )

    args = parser.parse_args()
    index_mapping, metadata = parse_video(args.video_path, 128, 72, 5)  # noqa: F841
    save_puzzle_video(
        "build/result.avi",
        Puzzle(index_mapping),
        metadata,
        128, 72, 5
    )


if __name__ == '__main__':
    main()
