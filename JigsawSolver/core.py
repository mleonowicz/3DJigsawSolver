import heapq
import pickle
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from JigsawSolver.dissimilarity import calculate_dissimilarity


class IndexToDataMapping:
    def __init__(
        self,
        n_pieces: Tuple[int, int, int],
        piece_size: Tuple[int, int, int],
        filename: str,
    ):
        """
        Mapping between the index of the piece and the video data.

        Parameters
        ----------
        n_pieces : Tuple[int, int, int]
            Number of pieces in a puzzle in each dimension
        piece_size : Tuple[int, int, int]
            Sizes of the pieces in each dimension
        filename : str
            Filename of the file that is being processed.
        """
        self.n_pieces_x, self.n_pieces_y, self.n_pieces_z = n_pieces
        self.num_of_puzzles = self.n_pieces_x * self.n_pieces_y * self.n_pieces_z
        self.width, self.height, self.depth = piece_size
        self.filename = filename

        if self.get_dissimilarity_cache_path().exists():
            with open(self.get_dissimilarity_cache_path(), "rb") as f:
                self.dissimilarity_cache = pickle.load(f)
        else:
            self.dissimilarity_cache = {}

        if self.get_dissimilarity_cache_path().exists():
            with open(self.get_dissimilarity_cache_path(), "rb") as f:
                self.best_fit_cache = pickle.load(f)
        else:
            self.best_fit_cache = {}

        self.id_map = {}
        for index in range(self.num_of_puzzles):
            self.id_map[index] = np.empty(
                (self.height, self.width, self.depth, 3), dtype=np.uint8
            )

    def get_dissimilarity_cache_path(self):
        return Path(
            f"{self.filename}_{str(self.n_pieces_x)}_{str(self.n_pieces_y)}_{str(self.n_pieces_z)}"
            + "_dissimilarity.cache"
        )

    def get_best_fit_cache_path(self):
        return Path(
            f"{self.filename}_{str(self.n_pieces_x)}_{str(self.n_pieces_y)}_{str(self.n_pieces_z)}"
            + "_best_fit.cache"
        )

    def save_caches(self):
        with open(self.get_dissimilarity_cache_path(), "wb+") as f:
            pickle.dump(self.dissimilarity_cache, f)

        with open(self.get_best_fit_cache_path(), "wb+") as f:
            pickle.dump(self.best_fit_cache, f)

    def coords_to_index(self, coords: Tuple[int, int, int]) -> int:
        """
        Mapping that converts given coordinates into a index of a puzzle.

        Parameters
        ----------
        coords : Tuple[int, int, int]
            (x, y, z) coordinates of the puzzle in the original image

        Returns
        -------
        int
            Index of the puzzle in the `id_map`
        """
        xcoord, ycoord, zcoord = coords
        return (
            xcoord * self.n_pieces_y * self.n_pieces_z
            + ycoord * self.n_pieces_z
            + zcoord
        )

    def index_to_coords(self, index: int) -> Tuple[int, int, int]:
        """
        Mapping that converts given index into coordinates of a puzzle.

        Parameters
        ----------
        index : int
            Index of the puzzle in the `id_map`

        Returns
        -------
        Tuple[int, int, int]
            Coordinates of the puzzle of id `index` in the original image
        """
        zcoord = index % self.n_pieces_z
        ycoord = (index // self.n_pieces_z) % self.n_pieces_y
        xcoord = (index // (self.n_pieces_z * self.n_pieces_y)) % self.n_pieces_x
        return (xcoord, ycoord, zcoord)

    def add_frame(self, frame: np.ndarray, frame_count):
        """
        Method used while building the mapping, it should be called for each frame in the video in order
        to correctly fill the data for all the puzzle pieces.

        Parameters
        ----------
        frame : np.ndarray
            Frame of the analyzed video
        frame_count : int
            Index of the frame in the video
        """
        zcoord, data_z_ind = frame_count // self.depth, frame_count % self.depth
        for xcoord, ycoord in product(range(self.n_pieces_x), range(self.n_pieces_y)):
            index = self.coords_to_index((xcoord, ycoord, zcoord))
            puzzle_data = self.id_map[index]
            puzzle_data[:, :, data_z_ind] = frame[
                self.height * ycoord : self.height * (ycoord + 1),
                self.width * xcoord : self.width * (xcoord + 1),
            ]

    def __getitem__(self, index):
        return self.id_map[index]

    def get_dissimilarity(self, index_a: int, index_b: int, orientation: str) -> float:
        """
        Calculates dissimilarity betweeen puzzles of `index_a` and `index_b` ids given an orientation.
        This function caches the calculated values.

        Parameters
        ----------
        index_a : int
            Index of the first puzzle to compare.
        piece_b : int
            Index of the second puzzle to compare.
        orientation : str
            Orientation of the puzzles. There are three valid arguments:
            * 'LR' = Left - Right
            * 'UD' = Up - Down
            * 'FB' = Forward - Back

        Returns
        -------
        float
            Calculated dissimilarity
        """
        key = (index_a, index_b, orientation)
        try:
            return self.dissimilarity_cache[key]
        except KeyError:
            dissimilarity = calculate_dissimilarity(
                self[index_a], self[index_b], orientation
            )
            self.dissimilarity_cache[key] = dissimilarity
            return dissimilarity

    def get_best_fit(self, index: int, orientation: str) -> List[tuple[int, float]]:
        """
        Parameters
        ----------
        index : int
            Index of the puzzle
        orientation : str
            Valid arguments are 'L', 'R', 'U', 'D', 'F' and 'B'.

        Returns
        -------
        List[tuple[int, float]]
            List of tuples (index of the puzzle, dissimilarity) sorted in an ascending order
        """
        key = (index, orientation)
        try:
            return self.best_fit_cache[key]
        except KeyError:
            self.best_fit_cache[key] = []
            for other_index in range(self.num_of_puzzles):
                if other_index == index:
                    continue

                if orientation == "R":
                    dissimilarity = self.get_dissimilarity(index, other_index, "LR")
                elif orientation == "L":
                    dissimilarity = self.get_dissimilarity(other_index, index, "LR")
                if orientation == "D":
                    dissimilarity = self.get_dissimilarity(index, other_index, "UD")
                elif orientation == "U":
                    dissimilarity = self.get_dissimilarity(other_index, index, "UD")
                if orientation == "B":
                    dissimilarity = self.get_dissimilarity(index, other_index, "FB")
                elif orientation == "F":
                    dissimilarity = self.get_dissimilarity(other_index, index, "FB")

                self.best_fit_cache[key].append((other_index, dissimilarity))

            self.best_fit_cache[key] = sorted(
                self.best_fit_cache[key], key=lambda t: t[1]
            )
            return self.best_fit_cache[key]


class Puzzle:
    def __init__(
        self, mapping: IndexToDataMapping, puzzle_pieces: np.ndarray | None = None
    ):
        """
        Representation of a single solution to the puzzle

        Parameters
        ----------
        mapping : IndexToDataMapping
            Mapping between the puzzle indexes and the video data corresponding to each piece
        puzzle_pieces : np.ndarray
            3D numpy array that with indices of puzzles
        """
        self.index_to_coord = {}
        self._fitness: float | None = None
        self.index_to_data = mapping
        self.n_x, self.n_y, self.n_z = (
            mapping.n_pieces_x,
            mapping.n_pieces_y,
            mapping.n_pieces_z,
        )
        if puzzle_pieces is not None:
            self.puzzle = puzzle_pieces
        else:
            # shuffling the pieces
            self.puzzle = np.arange(mapping.num_of_puzzles, dtype=np.int32)
            np.random.shuffle(self.puzzle)
            self.puzzle = np.reshape(self.puzzle, (self.n_x, self.n_y, self.n_z))

        for (x, y, z), index in np.ndenumerate(self.puzzle):
            self.index_to_coord[index] = (x, y, z)

    @property
    def fitness(self) -> float:
        """
        Fitness value getter that is calculated by summing dissimilarity between all adjecent pieces.

        Returns
        -------
        float
            Fitness value
        """
        if self._fitness:
            return self._fitness

        self._fitness = 0
        for i in range(self.n_x - 1):
            for j in range(self.n_y):
                for k in range(self.n_z):
                    index_a = self.puzzle[i][j][k]
                    index_b = self.puzzle[i + 1][j][k]
                    self._fitness += self.index_to_data.get_dissimilarity(
                        index_a, index_b, "LR"
                    )

        for i in range(self.n_x):
            for j in range(self.n_y - 1):
                for k in range(self.n_z):
                    index_a = self.puzzle[i][j][k]
                    index_b = self.puzzle[i][j + 1][k]
                    self._fitness += self.index_to_data.get_dissimilarity(
                        index_a, index_b, "UD"
                    )

        for i in range(self.n_x):
            for j in range(self.n_y):
                for k in range(self.n_z - 1):
                    index_a = self.puzzle[i][j][k]
                    index_b = self.puzzle[i][j][k + 1]
                    self._fitness += self.index_to_data.get_dissimilarity(
                        index_a, index_b, "FB"
                    )

        return self._fitness

    def get_adjecent_piece(self, piece_id: int, orientation: str) -> Optional[int]:
        try:
            x, y, z = self.index_to_coord[piece_id]
            if orientation == "R":
                return self.puzzle[x + 1, y, z]
            if orientation == "L":
                return self.puzzle[x - 1, y, z]
            if orientation == "U":
                return self.puzzle[x, y - 1, z]
            if orientation == "D":
                return self.puzzle[x, y + 1, z]
            if orientation == "F":
                return self.puzzle[x, y, z + 1]
            if orientation == "B":
                return self.puzzle[x, y, z - 1]
        except IndexError:
            pass
        return None


class CrossOperator(object):
    def __init__(
        self, first_parent: Puzzle, second_parent: Puzzle, mutation_probability=0.01
    ):
        self.first_parent = first_parent
        self.second_parent = second_parent
        self.mutation_probability = mutation_probability

        self.mapping = first_parent.index_to_data
        self.num_of_puzzles = first_parent.index_to_data.num_of_puzzles

        self.width = first_parent.n_x
        self.height = first_parent.n_y
        self.depth = first_parent.n_z

        self.left_b = 0
        self.right_b = 0
        self.up_b = 0
        self.down_b = 0
        self.forward_b = 0
        self.back_b = 0

        self.kernel: dict[Tuple[int, int, int], int] = {}
        self.piece_candidates: list = []

    def __call__(self):
        start_piece_id = np.random.choice(self.num_of_puzzles, 1)[0]
        start_piece_position = (0, 0, 0)

        self.available_pieces = set(range(self.num_of_puzzles))
        self.add_to_kernel(start_piece_id, start_piece_position)

        while len(self.kernel) != self.num_of_puzzles:
            _, (index, new_position, old_index, orientation) = heapq.heappop(
                self.piece_candidates
            )

            if new_position in self.kernel or not self.is_in_boundary(new_position):
                continue
            if index not in self.available_pieces:
                priority, new_piece_index = self.get_new_piece_index(
                    old_index, orientation
                )
                # if np.random.uniform() < self.mutation_probability:
                #     new_piece_index = np.random.choice(list(self.available_pieces), 1)[0]

                heapq.heappush(
                    self.piece_candidates,
                    (priority, (new_piece_index, new_position, old_index, orientation)),
                )
                continue

            self.add_to_kernel(index, new_position)

        return self.procreate()

    def add_to_kernel(
        self, current_piece_index: int, curr_piece_position: Tuple[int, int, int]
    ):
        curr_x, curr_y, curr_z = curr_piece_position
        self.kernel[curr_piece_position] = current_piece_index
        self.available_pieces -= set([current_piece_index])
        self.left_b = min(self.left_b, curr_x)
        self.right_b = max(self.right_b, curr_x)
        self.up_b = min(self.up_b, curr_y)
        self.down_b = max(self.down_b, curr_y)
        self.back_b = min(self.back_b, curr_z)
        self.forward_b = max(self.forward_b, curr_z)

        adjecent_positions = {
            "R": (curr_x + 1, curr_y, curr_z),
            "L": (curr_x - 1, curr_y, curr_z),
            "D": (curr_x, curr_y + 1, curr_z),
            "U": (curr_x, curr_y - 1, curr_z),
            "F": (curr_x, curr_y, curr_z + 1),
            "B": (curr_x, curr_y, curr_z - 1),
        }

        for orientation in ["R", "L", "D", "U", "B", "F"]:
            new_position = adjecent_positions[orientation]
            if new_position in self.kernel:
                continue

            if not self.is_in_boundary(new_position):
                continue

            if np.random.uniform() < self.mutation_probability:
                new_piece_index = np.random.choice(list(self.available_pieces), 1)[0]
                priority = 0.0
            else:
                priority, new_piece_index = self.get_new_piece_index(
                    current_piece_index, orientation
                )

            heapq.heappush(
                self.piece_candidates,
                (
                    priority,
                    (new_piece_index, new_position, current_piece_index, orientation),
                ),
            )

    def get_new_piece_index(
        self, current_piece_index: int, orientation: str
    ) -> Tuple[float | int, int]:
        # Same piece in both parents
        first_parent_adjecent_index = self.first_parent.get_adjecent_piece(
            current_piece_index, orientation
        )
        second_parent_adjecent_index = self.second_parent.get_adjecent_piece(
            current_piece_index, orientation
        )

        if (
            first_parent_adjecent_index is not None
            and second_parent_adjecent_index is not None
        ):
            if first_parent_adjecent_index in self.available_pieces:
                if first_parent_adjecent_index == second_parent_adjecent_index:
                    return -2, first_parent_adjecent_index

        # Best buddy in at least one parent
        best_fit_index, _ = self.mapping.get_best_fit(current_piece_index, orientation)[
            0
        ]
        if best_fit_index in self.available_pieces:
            inverse_best_fit_index, _ = self.mapping.get_best_fit(
                best_fit_index, self.get_inverse_orientation(orientation)
            )[0]

            if inverse_best_fit_index == current_piece_index:
                if best_fit_index in (
                    first_parent_adjecent_index,
                    second_parent_adjecent_index,
                ):
                    return -1, best_fit_index

        # Best available fit
        best_fits_list = self.mapping.get_best_fit(current_piece_index, orientation)
        for candidate_index, candidate_score in best_fits_list:
            if candidate_index in self.available_pieces:
                return candidate_score, candidate_index

        # Should not end up here
        assert False

    def get_inverse_orientation(self, orientation: str) -> str:
        if orientation == "R":
            return "L"
        if orientation == "L":
            return "R"
        if orientation == "D":
            return "U"
        if orientation == "U":
            return "D"
        if orientation == "F":
            return "B"
        if orientation == "B":
            return "F"
        raise KeyError

    def is_in_boundary(self, position: Tuple[int, int, int]) -> bool:
        x, y, z = position
        if self.right_b - self.left_b >= self.width - 1:
            if x < self.left_b or x > self.right_b:
                return False

        if self.down_b - self.up_b >= self.height - 1:
            if y > self.down_b or y < self.up_b:
                return False

        if self.forward_b - self.back_b >= self.depth - 1:
            if z > self.forward_b or z < self.back_b:
                return False

        return True

    def procreate(self) -> Puzzle:
        new_pieces = np.empty((self.width, self.height, self.depth))
        for pos, index in self.kernel.items():
            x, y, z = pos
            x = x - self.left_b
            y = y - self.up_b
            z = z - self.back_b
            new_pieces[x, y, z] = index
        return Puzzle(self.mapping, new_pieces)
