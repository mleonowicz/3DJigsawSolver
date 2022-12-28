import argparse

from JigsawSolver.video_utility import create_puzzle

if __name__ == "__main__":
    VIDEO_PATH = "build/example.mp4"
    puzzle = create_puzzle(VIDEO_PATH, 128, 72, 5)