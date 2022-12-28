import argparse
import sys

from JigsawSolver.video_utility import create_puzzle


def main(argv):
    parser = argparse.ArgumentParser(
        prog='3DJigsawSolver'
    )
    parser.add_argument(
        'video_path',
        type=str
    )

    args = parser.parse_args()
    puzzle = create_puzzle(args.video_path, 128, 72, 5)  # noqa: F841


if __name__ == '__main__':
    main(sys.argv)
