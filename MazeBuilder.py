"""
Module to construct and solve labyrinths.
"""

import matplotlib.pyplot as pyplot
import md5
import numpy as np
import random
import sys


class MazeBuilder(object):

    def __init__(self, N=250, random_start=False):
        """
        Create MazeBuilder for maze N x N.
        Starts carving at N/2, N/2 unless random_start is True.
        """
        self._lab_size = N
        self._maze = np.ones((N, N))
        # Set of the squares which should not be modified anymore
        self.visited = set()
        # Set of walls to consider for the passage
        self.walls = set()
        # Starting point
        if not random_start:
            middle = N / 2
            x_start = middle
            y_start = middle
        else:
            x_start = random.randint(0, N - 1)
            y_start = random.randint(0, N - 1)
        self._add_neighbours(x_start, y_start)
        self.visited.add((x_start, y_start))
        self._maze[x_start][y_start] = 255

        # Path through the labyrinth
        self._solution = None

        self._built = False

    def _add_wall(self, x, y):
        if (x, y) in self.walls:
            self.visited.add((x, y))
        else:
            self.walls.add((x, y))

    def _add_neighbours(self, x, y):
        if x > 0:
            self._add_wall(x - 1, y)
        if x < self._lab_size - 1:
            self._add_wall(x + 1, y)
        if y > 0:
            self._add_wall(x, y - 1)
        if y < self._lab_size - 1:
            self._add_wall(x, y + 1)

    def build(self):
        """
        Build the maze. Return true iff top left and bottom right corners are
        empty.
        """
        while len(self.walls) > 0:
            # Choose the random wall
            new = random.sample(self.walls, 1)[0]
            x, y = new
            # If it's not fixed, dig through it
            if not (x, y) in self.visited:
                self.visited.add((x, y))
                self._maze[x][y] = 255
                self._add_neighbours(x, y)
            # We are done with this wall
            self.walls.remove(new)

        self._built = True
        return self._maze[0][0] == 255 and \
            self._maze[self._lab_size - 1][self._lab_size - 1] == 255

    def plot(self,
             color_corners=True,
             show=True,
             save_name=None,
             figsize=(10, 10)):
        """
        Plot the built maze.

        If color_corners is True, color top left and bottom right corners.

        If show is True, call pyplot.show().

        If save_name is not None, save the pyplot picture under
        `save_name`.png name.

        Parameter figsize is passed to pyplot.figure.
        """
        assert self._built

        to_plot = np.copy(self._maze)

        if color_corners:
            to_plot[0][0] = 50
            to_plot[self._lab_size - 1][self._lab_size - 1] = 150

        pyplot.figure(figsize=figsize)
        pyplot.imshow(
            to_plot, cmap=pyplot.cm.spectral, interpolation='nearest')
        pyplot.xticks([]), pyplot.yticks([])

        if save_name:
            pyplot.savefig(save_name + '.png')

        if show:
            pyplot.show()

    def save_npy(self, save_name='maze'):
        """
        Save maze as numpy file; the name used is `save_name`.npy.
        """
        np.save(save_name + '.npy', self._maze)

    def save_csv(self, save_name='maze'):
        """
        Save maze as csv file; the name used is `save_name`.csv.
        """
        np.savetxt(save_name + '.csv', self._maze)

    def solve(self):
        """
        Find a path through labirynth; from top left corner to bottom right
        corner.
        """
        assert self._built
        assert self._maze[0][0] == 255 and self._maze[
            self._lab_size - 1][self._lab_size - 1] == 255

        # We'll do breadth search; path is an array filled with distances,
        # queue is the queue of the algorithm
        path = np.zeros_like(self._maze)
        queue = set()
        queue.add((0, 0))

        # Make a closure
        def mark_one(x, y, dist):
            self._mark_one(queue, path, x, y, dist)

        # XXX (0, 0) will get re-added to the queue, but it won't lead to any
        # consequences.
        while len(queue) > 0:
            nx, ny = queue.pop()
            dist = path[nx][ny]
            if nx > 0:
                mark_one(nx - 1, ny, dist)
            if nx < self._lab_size - 1:
                mark_one(nx + 1, ny, dist)
            if ny > 0:
                mark_one(nx, ny - 1, dist)
            if ny < self._lab_size - 1:
                mark_one(nx, ny + 1, dist)

        # Now backtrack to find the path
        self._solution = []
        x = self._lab_size - 1
        y = self._lab_size - 1
        max_dist = self._lab_size * self._lab_size
        path[x][y] = max_dist

        # Walls didn't get the distance, wrote there big number
        path[path == 0] = max_dist

        path[0][0] = 0

        while x != 0 or y != 0:
            self._solution.append((x, y))
            dist = path[x][y]
            # Move where the distance to the start is smaller than in the
            # current cell
            if x > 0:
                if path[x - 1][y] < dist:
                    x -= 1
                    continue
            if x < self._lab_size - 1:
                if path[x + 1][y] < dist:
                    x += 1
                    continue
            if y > 0:
                if path[x][y - 1] < dist:
                    y -= 1
                    continue
            if y < self._lab_size - 1:
                if path[x][y + 1] < dist:
                    y += 1
                    continue
            assert False, "Backtracking failed: program error"

    def _mark_one(self, queue, path, x, y, dist):
        """
        Perform one step of breadth search.
        queue -- the queue of search algorithm
        path -- filled with known distances
        x, y -- coordinates of cell to process
        dist -- current path length
        """
        # If the cell is not empty, do nothing
        if self._maze[x][y] != 255:
            return
        # If path to cell is found already, do nothing
        if path[x][y] != 0:
            return
        queue.add((x, y))
        path[x][y] = dist + 1

    def plot_solution(self, show=True, save_name=None, figsize=(10, 10)):
        """
        Plot the solution.

        If show is True, call pyplot.show().

        if save_name is not None, save the pyplot picture under
        `save_name`.png name.

        Parameter figsize is passed to pyplot.figure.
        """
        assert self._solution

        solved = np.copy(self._maze)
        solved[0][0] = 150
        for x, y in self._solution:
            solved[x][y] = 150

        pyplot.figure(figsize=figsize)
        pyplot.imshow(solved, cmap=pyplot.cm.spectral, interpolation='nearest')
        pyplot.xticks([]), pyplot.yticks([])

        if save_name:
            pyplot.savefig(save_name + '.png')

        if show:
            pyplot.show()

    def print_solution(self):
        """
        Print solution in terms of S/E/N/W directions and its md5 sum.
        """
        assert self._solution

        x, y = 0, 0
        s = ""

        while len(self._solution) > 0:
            nx, ny = self._solution.pop()
            if nx == x + 1:
                s = s + "S"
            if nx == x - 1:
                s = s + "N"
            if ny == y + 1:
                s = s + "E"
            if ny == y - 1:
                s = s + "W"
            x = nx
            y = ny
        print s
        m = md5.new(s)
        print m.hexdigest()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "%s <labyrinth_size> <labyrinth_name>" % sys.argv[0]
        sys.exit(-1)
    N = int(sys.argv[1])
    lab_name = sys.argv[2]
    success = False
    while not success:
        lab = MazeBuilder(N=N)
        success = lab.build()
        if not success:
            print "Corners are not empty, next try"
    print "Labyrinth built"
    lab.plot(save_name=lab_name)
    lab.save_npy(lab_name)
    lab.save_csv(lab_name)
    lab.solve()
    lab.plot_solution(save_name=lab_name + "_solution")
    lab.print_solution()
