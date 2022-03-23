from random import randint, uniform
from typing import List
import time

mutation_prob = 0.05
n_queen = 8

# class for board


class Board:
    # constructor
    def __init__(self, board: List[int] = None):
        self.prob = -1
        self.board: List[int] = self.init_chromosome(
        ) if board is None else board.copy()
        self.fit = self.calcFit()

    # build a chromosome where the index is the row and value is the column, there will be 0 column repeats
    def init_chromosome(self) -> List[int]:
        column_pool = [i for i in range(n_queen)]
        chromosome = []

        while len(column_pool) > 0:
            i = randint(0, len(column_pool)-1)
            chromosome.append(column_pool[i])
            column_pool.pop(i)

        return chromosome

    # calculate fitness for board

    def calcFit(self) -> float:
        sumFit = 0

        for i in range(n_queen-1):
            q1 = self.board[i]
            currFit = 0

            for j in range(i + 1, n_queen):
                d = j - i
                q2 = self.board[j]

                # check for queens in all directions
                if q1 == q2 or q1 == q2 + d or q1 == q2 - d:
                    currFit += 1
            sumFit += currFit

        return sumFit

    # create a copy of the board
    def copy(self) -> 'Board':
        return Board(self.board)

    # make a mutation by changing the position of the queen
    def mutate(self):
        if mutation_prob > uniform(0, 1):
            i = randint(0, n_queen-1)
            v = randint(0, n_queen-1)

            while self.board[i] is v:
                v = randint(0, n_queen-1)

            self.board[i] = v

        # calculate new fitness value
        self.fit = self.calcFit()


def roulette(brds: List[Board]):
    total_fitness = sum(1 / board.fit for board in brds)
    offset = 0

    for board in brds:
        board.prob = offset + 1 / board.fit / total_fitness
        offset = board.prob


def selectParents(boards: List[Board]) -> (Board, Board):
    # randomly sel;ect parents
    u1 = uniform(0, 1)
    u2 = uniform(0, 1)

    p1, p2 = None, None
    for board in boards:
        if board.prob > u1:
            p1 = board
            break

    for board in boards:
        if board.prob > u2:
            p2 = board
            break

    return p1, p2


def crossover(parent_1: Board, parent_2: Board) -> (Board, Board):
    # clone/create braeakpoint at random

    c1, c2 = parent_1.copy(), parent_2.copy()

    if 0.3 > uniform(0, 1):
        return c1, c2

    bp = randint(1, 6)

    c1.board[bp:], c2.board[bp:] = c2.board[bp:], c1.board[bp:]

    return c1, c2


def new_population(n: int) -> List[Board]:
    return [Board() for _ in range(n)]


if __name__ == '__main__':

    solutions = input("Enter the number of solutions to find: ")

    population_size = 30  # population size
    mutation_prob = 0.05  # mutation probability
    num_solutions = int(solutions)  # number of solutions wanted

    sol_set = set()

    boards = new_population(population_size)

    generations = 0

    if num_solutions > 0 and num_solutions < 93:
        print(f"Solutions wanted: {num_solutions}")
    else:
        print('Failed: Can only find 1-92 soltutions for 8 queens!')
        exit()

    if mutation_prob != 0.05:
        print(f"Mutation probability: {mutation_prob}")

    print(f"Starting population size: {population_size}")

    # start timer
    start = time.time()

    while True:
        generations += 1

        solution_found = False
        for board in boards:
            if board.fit == 0:
                print(generations, "generations later: Solution: ", board.board)
                sol_set.add(hash(tuple(board.board)))
                solution_found = True
                break

        # get new population if solution found or stop loop if all solutions found
        if solution_found:
            cur_num_solutions = len(sol_set)
            if cur_num_solutions == num_solutions:
                print("Current number of solutions:", cur_num_solutions)
                break

            print("Current number of solutions:", cur_num_solutions)
            boards = new_population(population_size)
            generations = 0

        roulette(boards)

        generation = []

        for _ in range(int(population_size/2)):
            p1, p2 = selectParents(boards)
            c1, c2 = crossover(p1, p2)

            c1.mutate()
            c2.mutate()

            generation.append(c1)
            generation.append(c2)

        boards = generation

    # end timer
    total_time = time.time()-start
    print(f"Finished after {total_time} ms")
