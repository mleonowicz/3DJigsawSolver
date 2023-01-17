import argparse

from JigsawSolver.core import Puzzle
from JigsawSolver.genetic_algorithm import GeneticAlgorithm
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
    parser.add_argument(
        '--pieces-number-x',
        type=int,
        default=10
    )
    parser.add_argument(
        '--pieces-number-y',
        type=int,
        default=10
    )
    parser.add_argument(
        '--pieces-number-z',
        type=int,
        default=10
    )

    args = parser.parse_args()
    index_mapping, metadata = parse_video(
        args.video_path,
        args.pieces_number_x,
        args.pieces_number_y,
        args.pieces_number_z
    )  # noqa: F841
    ga = GeneticAlgorithm(index_mapping, 100, 20)
    _, p = ga.fit(100)
    save_puzzle_video(
        "build/result.avi",
        p,
        metadata
    )


if __name__ == '__main__':
    main()
