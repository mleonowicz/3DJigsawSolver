import argparse

from JigsawSolver.video_utility import create_puzzle


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
    puzzle = create_puzzle(args.video_path, 128, 72, 5)  # noqa: F841


if __name__ == '__main__':
    main()
