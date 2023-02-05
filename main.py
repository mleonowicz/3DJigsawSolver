import argparse

from JigsawSolver.genetic_algorithm import GeneticAlgorithm
from JigsawSolver.video_utility import parse_input, save_puzzle_video


def main():
    parser = argparse.ArgumentParser(
        prog='3DJigsawSolver'
    )
    parser.add_argument(
        '--video-path',
        type=str,
        default='example/example-2.mp4',
    )
    parser.add_argument(
        '--pieces-number-x',
        type=int,
        default=15
    )
    parser.add_argument(
        '--pieces-number-y',
        type=int,
        default=15
    )
    parser.add_argument(
        '--pieces-number-z',
        type=int,
        default=15
    )
    parser.add_argument(
        '--input-type',
        choices=['video', 'image'],
        default='video'
    )
    parser.add_argument(
        '--alpha',
        default=0.003,
        type=float
    )
    parser.add_argument(
        '--beta',
        default=0.05,
        type=float
    )
    parser.add_argument(
        '--elites',
        default=15,
        type=int
    )
    parser.add_argument(
        '--population-size',
        default=500,
        type=int
    )
    parser.add_argument(
        '--max-iterations',
        default=100,
        type=int
    )
    parser.add_argument(
        '--output-path',
        default='output.dat',
        type=str
    )

    args = parser.parse_args()
    index_mapping, metadata = parse_input(
        args.video_path,
        args.pieces_number_x,
        args.pieces_number_y,
        args.pieces_number_z,
        args.input_type
    )
    ga = GeneticAlgorithm(
        index_mapping,
        population_size=args.population_size,
        elites=args.elites,
        alpha=args.alpha,
        beta=args.beta,
        output_path=args.output_path
    )
    _, p = ga.fit(args.max_iterations)
    save_puzzle_video(
        'result.avi',
        p,
        metadata
    )
    index_mapping.save_caches()


if __name__ == '__main__':
    main()
