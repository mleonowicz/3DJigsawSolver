# 3D Jigsaw Solver - Report

## Problem definition

In our project we are using the genetic algorithm methods for solving the jigsaw puzzles.
In the [paper](https://arxiv.org/pdf/1711.06767.pdf) paper the method was shown for solving the 2D puzzles.
We are generalizing the solution for the 3D case.

We have $n$ three dimensional puzzle pieces, that are constructed by splitting the input video/3D image.
Goal is to reconstruct the original input by exploring the space of all possible puzzle permutations.

This problem is known to be NP-hard.
We use a genetic algorithm with custom mutation and crossover algorithm.


## Used paper - https://arxiv.org/pdf/1711.06767.pdf

Our implementation is based on a [paper](https://arxiv.org/pdf/1711.06767.pdf) that proposes a solution for 2D puzzles.
We expand on the ideas introduced there.

## Implementation

We take an input - either 3D image, or a video (in that case the time will be considered as third dimension). We split it into the $n$ identically shaped puzzles to form a puzzle of $W \times H \times Z$ shape.
We then follow the genetic algorithm of the form:
* Create an initial population - each invididual has randomly placed puzzle pieces.
* For each generation:
    * Evaluate the cureent population.
    * Pick $N_E$ elites that remain in the population
    * We randomly choose with weights based on fitness two members of population and cross the to create a new member. We repeat that until the population is full.
    * With $\beta$ probability we perform mutation of the new member in a procedure described below.
* Alogrithm returns puzzle solution with the best fitness score.

### Fitness

Lets define the dissimilarity function between two puzzle pieces $x_i$ and $x_j$ ($x_i, x_j$ are of size $W \times H \times Z$). We compare them in the front-back direction, i. e. $x_j$ is the backward piece and $x_i$ is the frontal one.

$$D_Z(x_i, x_j, r) = \sqrt{\sum_{w=1}^W \sum_{h=1}^H \sum_{c=1}^3 (x_i(w, h, Z, c) - x_j(w, h, 1, c))^2}$$

$D_H, D_W$ are dissimilarity functions for the up-down and left-right directions respectively and are defined analogously.

Intuitively, we calculate the difference between adjecent borders of two pieces (in rgb format) that are next to each other.
It is based on a premise, that adjecent pieces in the original puzzle should share similar coloring.

Fitness function of the entire puzzle solution is the sum over dissimilarity measures between all pairing of neighbour pieces in all three directions.

### Crossover

During the crossover operation we maintain a kernel - set of pieces with relative position saved.
We add new pieces to the kernel iteratively by taking the best fit piece for the current configuration in a greedy manner.
The exact position of each piece is set at the final step of the kernel creation, when all puzzle pieces are added into the kernel.
The intuition behind the operation is that when we have a matching pair of pieces, it should be possible to shift the around 'the board'.
We do not want to fix the puzzle piece into a single place, but rather we place the piece based on its position relative to other pieces.

Crossover operator definition:
* Start by adding a random piece into the kernel
* Every time a new puzzle is added, check each of its borders that is not taken (left, right, up, down, forward, backward) and for each of those borders calculate which piece is the best fit for that place using the following steps.
    * If both parents have the the same puzzle adjecent to the piece under consideration and this puzzle is in the available pieces it is used and added to the kernel.
    * If the added puzzle has a best-buddy puzzle in the available piece and it is an adjecent puzzle in at least one of the parents then it is added to the kernel.
    * Puzzle that is the best fitting puzzle from the available pieces is added to the kernel.

Pieces $P_1$ and $P_2$ are best-buddy pieces if for one of $P_1$'s borders $P_2$ has the smallest dissimilarity, and for corresponding border of $P_2$ the best fit is $P_1$.

TODO: Add visualization video 

### Mutation

We introduce two types of mutation:
* In crossover operator when calculating the best-fit puzzle a randomly chosen puzzle from the available pieces is picked and added to the kernel instead with probability $\alpha$.

* After the crossover operator with probability $\beta$ we will perform mutation by reversing the order of the puzzle placement in the third dimension (depth). Lets consider layers of puzzle pieces, placed along the third dimension. Two indices are picked $0 \leq i \leq j \leq Z$ and then puzzles layers between the $i$ and $j$ are inversed.
For example if $i = 2$ and $j = 5$ then pieces of depth $2$ swap with pieces of depth $5$ and pieces with depth $3$ swap with pieces of depth $4$.

## Results

We have tested the image on two videos and three 3D images of medical scans.

TODO: wstawić oryginalne video, scans

Each experiment was run wil following parameters:
* Population number: 500
* Number of puzzles: 3375 - $15 \times 15 \times 15$ puzzles in each dimension
* Number of elites: 15
* $\alpha$: 0.003
* $\beta$: 0.05
* Maximum number of generations: 100

Results for each 

### Ablation study

## Further Work

TODO: Lepsze metryki