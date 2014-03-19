# -*- coding: utf-8 -*-
"""
Implementation of the 1959 Dijkstra algorithm for identifying the lowest-
cost path through a network.
"""

from __future__ import division

__author__ = "Branden Kappes"
__copyright__ = "Copyright 2014, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Branden Kappes"
__email__ = "branden.kappes@gmail.com"
__status__ = "Developmental"
__date__ = "Jan 10, 2014"


import inspect
import numpy as np
from bitarray import bitarray
from itertools import permutations, product
from operator import lt
from pymatgen.core.structured_grid import StructuredGrid
from heapq import heapify, heappush, heappop, cmp_lt

# try:
#     from numba import autojit, jit
# except ImportError:
#     # create a replacement decorators that do nothing
#     # if numba autojit and jit are not available
#     def autojit(f):
#         return f
#     class jit(object):
#         def __init__(self, *args, **kwds):
#             pass
#         def __call__(self, f):
#             return f
# 
# # Why not use bisect and insort from the bisect module? Because I need
# # to be able to specify the comparison function. By default, this orders
# # in ascending order, just like bisect.bisect and bisect.insort
# #@autojit
# def bisect(a, x, comparable=lt):                    # pylint: disable=C0103
#     """
#     --------
#     Synopsis
#     --------
#     Finds the index where x would be inserted into a. If x is
#     already present in a, then index will be inserted to the
#     right, i.e. higher index.
# 
#     ----------
#     Parameters
#     ----------
#     :a (list): List sorted by cmp.
#     :x (object): Object whose position is to be identified.
#     :cmp (callable): Comparison that is to be used to identify the
#                      location of the x in a.
#     :returns (int): Index where x is to be inserted
#     """
#     # pylint: disable=C0103
#     lo = 0
#     hi = len(a)
#     while lo < hi:
#         mid = (lo+hi)//2
#         if comparable(x, a[mid]):
#             hi = mid                            #
#         else:
#             lo = mid+1
#     return lo
# 
# #@autojit
# def insort(a, x, comparable=lt):                     # pylint: disable=C0103
#     """
#     --------
#     Synopsis
#     --------
#     Insert item x in list a, and keep it sorted according
#     to cmp, assuming a is already so sorted.
# 
#     If x is already in a, insert it to the right of the
#     rightmost x.
# 
#     ----------
#     Parameters
#     ----------
#     :a (list): List to be sorted
#     :x (object): object that is to be inserted into a
#     :cmp (callable): comparable to apply when sorting the list
#     :returns: None
#     """
#     # pylint: disable=C0103
#     i = bisect(a, x, comparable)
#     a.insert(i, x)


class DijkstraPath(object):
    """
    Object that constructs and manages the lowest cost path through
    a StructuredGrid object.
    
    Methods
    -------
            
    .. method::
    
        cost: Returns an ndarray that contains the cost to reach
            any point from the start. This has the same dimensions
            as the grid object.
    
    .. method::
    
        path_to: Returns the path connecting the start point to the
            requested end point (see path_to's docstring).
    
    Setup attributes
    ----------------
    .. attribute::
    
        grid (StructuredGrid): Grid through which the path is to be forged.
            Set grid.is_periodic to control whether the path can move
            across any or all of the three periodic boundaries. Changing the
            grid will invalidate the path. The grid will not be changed by
            this DijkstraPath object.
    
    .. attribute::
    
        start (tuple of ints or tuple of floats): Starting point for
            the path as either a voxel index (int) or coordinate (float).
            Changing the start point initializes reconstruction.
    
    .. attribute::
    
        connections (iterable of 3-element iterables of int): Defines
            how voxels are connected. Default: ((0, 0, 1), (0, 0, -1),
            (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)), i.e. up, down,
            right, left, front, back for z up, y right, and x out of plane.
            Changing the connections invalidates the path and resets the
            start point.
    
    .. attribute::
    
        cost_function (function): A function (or other callable object)
            that calculates the cost of moving from one voxel (center) to
            another voxel (neighbor). This function must have the signature
            :code:`rval func(center, neighbor)` where both the less-than
            comparison operator and the addition operator are defined for
            the return value type. *IMPORTANT* The cost_function must
            be everywhere positive!
    """
    
    def __init__(self, grid, start=None, connections=None, costFunc=None):
        """
        Constructs a new DijkstraPath object through a specified grid.
        A start point must be provided before the Dijkstra algorithm can
        be evaluated.
        
        Args::
        
            grid (StructuredGrid, required): StructuredGrid object
            
            start (3-element iterable): The start point for the path, as
                axis-aligned indices (ints). (c.f. StructuredGrid.index
                to convert a cartesian coordinate to the corresponding index.)
            
            connections (iterable of 3-element iterables of int): Defines
                how voxels are connected.
            
            costFunc (function): Function (signature,
                :code:`rval func(center, neighbor)`) that calculates the cost
                of moving from one voxel (center, index) to the next
                (neighbor, index).
        """
        # initialize
        self._grid = None # reference to the StructuredGrid object to traverse
        self._cost = None # cost to get to each point from the given start 
        self._pred = None # predecessor to each point
        self._connections = None
        self._costFunction = None
        self._start = None
        # act on parameters
        self.grid = grid
        self.connections = connections
        self.cost_function = costFunc
        self.start = start
        
    def _adjacent(self, ijk):
        """
        Yields the index adjacent to ijk based on the connectivity of
        self.connections and the periodicity of the grid."""
        for cxn in self.connections:
            # adjacent index
            lmn = np.add(ijk, cxn)
            # is this index bounded, i.e. in the box?
            bounded = np.array((0 <= lmn) & (lmn < self.grid.shape))
            # if not, is the grid either bounded or periodic in that direction?
            if not np.all(bounded | self.grid.is_periodic):
                continue
            # if we get here, then return the index
            lmn %= self.grid.shape
            yield tuple(lmn)
    
    @property
    def grid(self):
        """Returns the grid object to move through"""
        return self._grid
    
    @grid.setter
    def grid(self, grid):
        """
        Sets the grid object to move through. All previous data is
        invalidated, including the cost function."""
        self._grid = grid
        self._cost = np.ndarray(grid.shape, dtype=float)
        self._pred = np.ndarray(grid.shape + (3,), dtype=int)
        self.start = None
        self.cost_function = None
    
    @property
    def cost(self):
        """
        Returns a numpy array that stores the cost of getting
        to any point in the grid from the specified start point.
        This should be used for read-only access, but it so large
        that returning a copy is unwieldy.  
        """
        return self._cost
    
    def path_to(self, end):
        """
        Yields the indices of the path to get from the start to the
        requested end point.
        
        Args::
        
            end (3-element iterable of ints): Axis-aligned index of the
                end point.
        
        Returns::
        
            Yields each point along the path, from start to end (inclusive).
            
        Example::
        
            :code:`foo = [self.grid[ijk] for ijk in self.path_to(end)]`
            
            *foo* holds the values from the grid along the path to *end*.
        """
        path = [tuple(end)]
        while self._pred[path[-1]] != self.start:
            path.append(self._pred[path[-1]])
        path.append(self.start)
        for ijk in reversed(path):
            yield ijk
    
    @property
    def connections(self):
        """
        Returns the connections; that is, how the grid is traversed to
        search for the path."""
        return self._connections

    @connections.setter
    def connections(self, cxns):
        """
        Sets the connections that define how the grid is to be traversed.
        
        Args::
        
            cxn (tuple of 3-element tuples of ints): Steps to go from one
                voxel to the next in traversing the grid to find the path.
        """
        if cxns is None:
            self._connections = np.array(((0, 0, 1), (0, 0, -1),
                                          (0, 1, 0), (0, -1, 0),
                                          (1, 0, 0), (-1, 0, 0)), dtype=int)
        else:
            self._connections = np.array(tuple(tuple(int(c) for c in cxn) \
                                               for cxn in cxns), type=int)
    
    @property
    def cost_function(self):
        """Gets the cost function"""
        return self._costFunction
    
    @cost_function.setter
    def cost_function(self, func):
        """
        Sets the cost function. To avoid unnecessary evaluation, set the
        start point to None before setting the cost function. The default
        cost function ensures all values are positive, but does not affect
        the values stored in the grid.
        
        Args::
        
            func (function): the function to use in stepping from index
                i to index j, where i and j are 3-element tuples of ints.
        """
        if func is not None:
            funcArgs = inspect.getargspec(func)
            # check that the function takes two arguments (len(funcArgs[0]) == 2),
            # or that the function takes a variable number of arguments,
            # funcArgs[1] is not None
            if not (len(funcArgs[0]) == 2 or funcArgs[1]):
                raise ValueError("Function signature must be func(i, j)")
            self._costFunction = func
        else:
            minval = np.min(self.grid)
            self._costFunction = lambda i, j: self.grid[j] - minval
        self.eval()
        
    @property
    def start(self):
        """Gets the starting point, as an index in the grid."""
        return self._start
    
    @start.setter
    def start(self, ijk):
        """
        Sets the start point.
        
        Args::
        
            ijk (iterable of int): 3-element iterable that contains the
                index of the starting point.
        """
        self._start = ijk
        self.eval()
    
    def eval(self):
        """Evaluate the Dijkstra algorithm."""
        self._heap()
        
    def _heap(self):
        """
        Use a heap to evaluate the Dijkstra algorithm.
        
        From http://www.cse.ust.hk/~dekai/271/notes/L10/L10.pdf
        Dijkstra(G, w, s) {
            for (each u \in V) {
                d[u] = infty
                color[u] = white
            }
            d[s] = 0
            pred[s] = NULL
            Q = (queue with all vertices)
            while (non_empty(Q)) {
                u = extract_min(Q)
                for (each v \in Adj(u)) {
                    if (d[u] + w(u, v) < d[v]) {
                        d[v] = d[u] + w(u, v)
                        decrease_key(Q, v, d[v])
                        pred[v] = u
                    }
                }
                color[u] = black
        }
        
        Args::
        
            None
        
        Returns::
        
            True if the path was completed successfully. False otherwise.
        """
        def decrease_key(heap, second, newitem):
            """
            For a heap containing a entries that are tuples such that
            :code:`first, second = (float, (int, int, int))` search for a
            matching second entry, then decrease the key for that entry.
            Except for changes to accommodate the desired inputs, this is
            taken directly from heapq._siftdown.
            
            Returns: None
            """
            # find the index corresponding to the second value
            for pos, entry in enumerate(heap):
                if entry[1] == second:
                    break
            heap[pos] = newitem
            # Follow the path to the root, moving parents down until finding
            # a place newitem fits.
            while pos > 0:
                parentpos = (pos - 1) >> 1
                parent = heap[parentpos]
                if cmp_lt(newitem, parent):
                    heap[pos] = parent
                    pos = parentpos
                    continue
                break
            heap[pos] = newitem
        # ensure all parameters have been set
        if self.grid is None:
            return False
        if self.start is None:
            return False
        # local copies/references to data
        cost = self._cost
        func = self.cost_function
        grid = self.grid
        pred = self._pred
        start = self.start
        # reset cost and predecessor arrays
        cost.flat[:] = np.infty
        cost[start] = 0.0
        pred.flat[:] = 0
        pred[start] = start # pred[ijk] != ijk except at start
        # no one is complete
        complete = np.ndarray(grid.shape, dtype=bool)
        complete.flat[:] = False
        queue = [(cost[ijk], ijk) for ijk in
                 product(map(range, self.grid.shape))]
        
        heapify(queue)
        while queue:
            # cost, indices
            c, ijk = heappop(queue)
            for lmn in self._adjacent(ijk):
                if complete[lmn]: continue
                vcost = cost[lmn] + func(ijk, lmn)
                if vcost < cost[lmn]:
                    cost[lmn] = vcost
                    decrease_key(queue, lmn, vcost)
                    pred[lmn] = ijk
            complete[ijk] = True

    #@autojit
#     def to_end(self, end):
#         """
#         Evaluate to find the lowest cost path.
#         
#         Args::
#         
#             end (3-element tuple of ints): The point at which to end.
#                 
#         Returns::
#         
#             ((cost, total_cost), path indices) if successful. False otherwise
#         """
#         # ensure all parameters have been set
#         if self.grid is None:
#             return False
#         if self._start is None:
#             return False
#         # For N = the stepsize in connections, and N != 1, but
#         # end - start != iN, with i an integer, then no direct path to the
#         # end node exists, though it could wrap periodically to reach the end.
#         # Evaluation will take a long time, until all accessible boxes
#         # have been checked, but that is unavoidable.
#         #
#         # if there is a problem with these closures (nested functions)
#         # something like "ERROR (variable) must have a single type" or
#         # "(var) is a cell variable", then see
#         # http://numba.pydata.org/numba-doc/dev/pythonstuff.html
#         #@jit('b1(int_[:], int_[:])')
#         def path_cmp(i, j):
#             """Comparable to use when comparing path values."""
#             # path_value must be set in the parent namespace before this
#             # function is called.
#             i = ijk_to_i(i)
#             j = ijk_to_i(j)
#             return path_value[i] > path_value[j]
#         
#         #@jit('int_[:,:](int_[:])')
#         def get_adjacent(center):
#             """
#             --------
#             Synopsis
#             --------
#             Returns a tuple of indices for adjacent nodes that have yet to be
#             considered central nodes.
#     
#             ----------
#             Parameters
#             ----------
#             :center (tuple of ints): Index of the central node whose
#                 adjacent nodes are sought.
#             :returns (tuple of tuple of ints): Indices of adjacent nodes that
#                 have not yet been central nodes.
#             """
#             dim = np.array(self.grid.shape, dtype=int)
#             periodic = self.grid.is_periodic
#             adjacent = []
#             ijk = np.zeros_like(center)
#             for dijk in self.connections:
#                 # axis-aligned indices for the prospective adjacent node
#                 ijk = center + dijk
#                 # adjust only those directions for which periodicity, set
#                 # by the grid itself, is True. This is the function:
#                 #
#                 #   ijk = ijk - n*(i // n)
#                 #
#                 # applied only to those directions that are periodic
#                 ijk[periodic] = ijk[periodic] - \
#                                 dim[periodic]*(ijk[periodic] // dim[periodic])
#                 # periodic searching was handled immediately above,
#                 # t.f. exclude potential nodes that lie beyond the boundaries
#                 # of the grid
#                 if np.all(0 <= ijk) and np.all(ijk < dim):
#                     # axis-aligned to flattened index
#                     i = ijk_to_i(ijk)
#                     # Exclude node if it has already been a center.
#                     # path_set must be defined in the parent namespace.
#                     if not path_set[i]:
#                         adjacent.append(tuple(ijk))
#             return np.array(adjacent)
#         # end 'def get_adjacent(center_node_index):'
#         
#         #@jit('int_(int_[:])')
#         def ijk_to_i(ijk):
#             """
#             Returns the flattened index *i* corresponding to the axis-aligned
#             index *ijk*, i.e. self.grid.flat[i] === self.grid[ijk]
#             """
#             dim = self.grid.shape
#             stride = np.array([dim[-1]*dim[-2], dim[-1], 1]) # C-style
#             #stride = np.array([1, dim[0], dim[0]*dim[1]]) # F-style
#             return np.sum(stride*ijk)
#     
#         # shortest path to each node, None if node not yet in path
#         # *NOTE* this variable is used in get_adjacent, so
#         # though it is declared here, it is referenced, though not
#         # used, above.
#         path_value = self.grid.size*[None]
#         # index to the previous node in the shortest path to each node
#         prev_node = self.grid.size*[None]
#         # bitarray holds those nodes that have (1) or have not (0) been
#         # considered as central nodes, i.e. nodes whose path has been
#         # established. *NOTE* this variable is used in get_adjacent, so
#         # though it is declared here, it is referenced, though not
#         # used, above.
#         path_set = bitarray(self.grid.size)
#         try:
#             path_set.setall(0)                          # pylint: disable=E1101
#         except AttributeError:
#             for i in xrange(len(path_set)):
#                 path_set[i] = False
#         # queue of node indices to serve as the central node, these
#         # are the adjacent nodes with lowest value from previous iterations
#         queue = [self.start]
#         # keep vector sorted in descending order relative to the value
#         # stored in path_value, so that the last has the shortest path
#     
#         # ----------- FORWARD ----------- #
#         central = queue.pop()
#         icentral = ijk_to_i(central)
#         prev_node[icentral] = None # reverse linked list
#         path_value[icentral] = 0.0
#         path_set[icentral] = 1
#         while np.any(central != end):
#             for adjacent in get_adjacent(central):
#                 iadjacent = ijk_to_i(adjacent)
#                 # calc function to get to adjacent from central
#                 path = self.cost_function(central, adjacent)
#                 if path < 0:
#                     raise ValueError("A negative value for the cost function "
#                                      "was encountered, but it must be "
#                                      "everywhere positive. May I suggest "
#                                      "f(i,j) --> exp(f(i,j)), or similar?")
#                 path += path_value[icentral]
#                 # has a path to adjacent already been found?
#                 if path_value[iadjacent] is not None:
#                     # is the new path lower cost?
#                     if path < path_value[iadjacent]:
#                         # remove adjacent node from previous location
#                         # in queue. Why not check for 0? Because
#                         # if we get here, adjacent will already be in
#                         # queue, and bisect, which returns the
#                         # right insertion point, will return a value in
#                         # the range [1,len(queue)]
#                         i = bisect(queue,
#                                    adjacent,
#                                    comparable=path_cmp)-1
#                         adjacent = queue.pop(i)
#                         iadjacent = ijk_to_i(adjacent)
#                     # no, then move to the next adjacent node
#                     else:
#                         continue
#                 # (re)add adjacent to the queue
#                 prev_node[iadjacent] = central
#                 path_value[iadjacent] = path
#                 insort(queue,
#                        adjacent,
#                        comparable=path_cmp)
#             # get the next central node, i.e. the adjacent node with the
#             # cheapest path, as there is no cheaper way to get to this
#             # soon-to-be-central node.
#             central = queue.pop()
#             icentral = ijk_to_i(central)
#             path_set[icentral] = 1
#         # ----------- BACKWARD ----------- #
#         pathway = []
#         cost = [] # cost at each point
#         total_cost = path_value[ijk_to_i(central)]
#         while central is not None:
#             pathway += [central]
#             cost += path_value[ijk_to_i(central)] # incremental cost
#             central = prev_node[central]
#         pathway.reverse()
#         cost.reverse()
#         # x = (cumx1, cumx2, cumx3, ..., cumxN) - (0, cumx1, cumx2, ...)
#         cost = tuple(np.array(cost) - np.array([0.0] + cost[:-1]))
#         # save the path and cost
#         return ((cost, total_cost), tuple(pathway))
#end class DijkstraPath
