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
from itertools import permutations
from operator import lt
from pymatgen.core.structured_grid import StructuredGrid

try:
    from numba import autojit, jit
except ImportError:
    # create a replacement decorators that do nothing
    # if numba autojit and jit are not available
    def autojit(f):
        return f
    class jit(object):
        def __init__(self, *args, **kwds):
            pass
        def __call__(self, f):
            return f

# Why not use bisect and insort from the bisect module? Because I need
# to be able to specify the comparison function. By default, this orders
# in ascending order, just like bisect.bisect and bisect.insort
#@autojit
def bisect(a, x, comparable=lt):                    # pylint: disable=C0103
    """
    --------
    Synopsis
    --------
    Finds the index where x would be inserted into a. If x is
    already present in a, then index will be inserted to the
    right, i.e. higher index.

    ----------
    Parameters
    ----------
    :a (list): List sorted by cmp.
    :x (object): Object whose position is to be identified.
    :cmp (callable): Comparison that is to be used to identify the
                     location of the x in a.
    :returns (int): Index where x is to be inserted
    """
    # pylint: disable=C0103
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if comparable(x, a[mid]):
            hi = mid                            #
        else:
            lo = mid+1
    return lo

#@autojit
def insort(a, x, comparable=lt):                     # pylint: disable=C0103
    """
    --------
    Synopsis
    --------
    Insert item x in list a, and keep it sorted according
    to cmp, assuming a is already so sorted.

    If x is already in a, insert it to the right of the
    rightmost x.

    ----------
    Parameters
    ----------
    :a (list): List to be sorted
    :x (object): object that is to be inserted into a
    :cmp (callable): comparable to apply when sorting the list
    :returns: None
    """
    # pylint: disable=C0103
    i = bisect(a, x, comparable)
    a.insert(i, x)


class DijkstraPath(object):
    """
    Object that constructs and manages the lowest cost path through
    a StructuredGrid object.
    
    Methods
    -------
            
    .. method::
    
        path (tuple of 3-element tuples of ints): the path (including the
            endpoints, start and end) that minimizes the cost function.
    
    .. method::
    
        cost: The cost, at each point, to traverse the path.
    
    .. method::
    
        total_cost: The total cost incurred in traversing the path.
    
    .. method::
    
        up_to_date(): Returns True if the path is up to date or False if
            a parameter that might change the path has been changed. Note:
            This does not check if the StructuredGrid has changed, which
            is likely to invalidate any existing path.
    
    Setup attributes
    ----------------
    .. attribute::
    
        grid (StructuredGrid): Grid through which the path is to be forged.
            Set grid.is_periodic to control whether the path can move
            across any or all of the three periodic boundaries.
    
    .. attribute::
    
        start (tuple of ints or tuple of floats): Starting point for
            the path as either a voxel index (int) or coordinate (float).
    
    .. attribute::
    
        end (tuple of ints or tuple of floats): End point for the path
            as either a voxel index (int) or coordinate (float).
    
    .. attribute::
    
        connections (iterable of 3-element iterables of int): Defines
            how voxels are connected. Default: ((0, 0, 1), (0, 0, -1),
            (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)), i.e. up, down,
            right, left, front, back for z up, y right, and x out of plane.
    
    .. attribute::
    
        cost_function (function): A function (or other callable object)
            that calculates the cost of moving from one voxel (center) to
            another voxel (neighbor). This function must have the signature
            :code:`rval func(center, neighbor)` where both the less-than
            comparison operator and the addition operator are defined for
            the return value type. *IMPORTANT* The cost_function must
            be everywhere positive!
    """
    
    def __init__(self, grid, startEnd=None,
                 connections=None, costFunc=None):
        """
        Constructs a new DijkstraPath object through a specified grid.
        All arguments, except the StructuredGrid object, are optional to
        instantiate a DijkstraPath object, but all are necessary before
        the path can be calculated.
        
        Args::
        
            grid (StructuredGrid, required): StructuredGrid object
            
            startEnd (2-element iterable of 3-element iterables): The start
                and end points for the path, as either indices (ints) or
                as coordinates (floats).
            
            connections (iterable of 3-element iterables of int): Defines
                how voxels are connected.
            
            costFunc (function): Function (signature,
                :code:`rval func(center, neighbor)`) that calculates the cost
                of moving from one voxel (center, index) to the next
                (neighbor, index).
        """
        # initialize output
        self._cost = (None, None) # (cost, total)
        self._path = None # (start, ijk1, ijk2, ..., end)
        # initialize input
        self.grid = grid
        self._connections = None
        self._costFunction = None
        self._end = None
        self._start = None
        self._upToDate = False
        # act on parameters
        self.connections = connections
        if costFunc is None:
            self.cost_function = lambda i, j: self.grid[j] - np.min(self.grid)
        else:
            self.cost_function = costFunc
        if startEnd:
            self.end = startEnd[1]
        if startEnd:
            self.start = startEnd[0]
    
    def _smart_point(self, point, type="auto"):
        """
        Conditions a point for use as a start/end point, performing a
        conversion, if necessary, from a point in Cartesian space to
        an index.
        
        Args::
        
            point (iterable of int-like or float-like): 3-element iterable
                that contains the starting point.
            
            type (string, "auto"|"index"|"coord"): If type is "auto" and
                point is int-like, then the point is taken as an index,
                otherwise as a coordinate in cartesian space.
        
        Returns::
            
            An nd.array (i, j, k), accounting for the periodicity 
            specified in DijkstraPath.grid or None, if *point* is None. 
        """
        #=======================================================================
        # Developer's Note
        # ----------------
        # This method might be better served as a public method so a user
        # could make use of the *type* keyword. However, I don't see a
        # user preconditioning their points.
        #=======================================================================
        # Allow a point to be None
        if point is None:
            return None
        # what type of point are we working with?
        type = type.lower()
        if type == "auto":
            isIndex = isinstance(point[0], (int, long, np.int, np.int0,
                                            np.int8, np.int16, np.int32,
                                            np.int64))
        elif type == "index":
            isIndex = True
        elif type == "coord":
            isIndex = False
        else:
            raise ValueException("%s is not a valid point type, please select "
                                 "from 'auto', 'index', or 'coord'" % type)
        # Ensure the point is a triplet of indices
        if not isIndex:
            ijk = np.array(self.grid.index(point), copy=True)
        else:
            ijk = np.array(point, dtype=int, copy=True)
        # follow the periodicity of the underlying grid
        dim = np.asarray(self.grid.shape, dtype=int)
        periodic = self.grid.is_periodic
        try:
            ijk[periodic] = ijk[periodic] - \
                            dim[periodic]*(ijk[periodic] // dim[periodic])
        except:
            print "dim:", dim
            print "dim[periodic]:", dim[periodic]
            print "periodic:", periodic
            print "ijk:", ijk
            print "ijk[periodic]:", ijk[periodic]
            print "ijk[periodic] // dim[periodic]:", \
                ijk[periodic] // dim[periodic]
            raise
        return ijk
    
    def cost(self):
        """Returns the cost incurred in traversing the path."""
        return self._cost[0]
    
    def path(self):
        """Returns the path taken (as indices) through the grid."""
        return self._path
    
    def total_cost(self):
        """
        Returns the total cost of traversing the path, i.e. the sum
        of self.cost.
        """
        return self._cost[1]
    
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
        self._upToDate = False
    
    @property
    def cost_function(self):
        """Gets the cost function"""
        return self._costFunction
    
    @cost_function.setter
    def cost_function(self, func):
        """Sets the cost function"""
        funcArgs = inspect.getargspec(func)
        # check that the function takes two arguments (len(funcArgs[0]) == 2),
        # or that the function takes a variable number of arguments,
        # funcArgs[1] is not None
        if not (len(funcArgs[0]) == 2 or funcArgs[1]):
            raise ValueError("Function signature must be func(i, j)")
        self._costFunction = func
        self._upToDate = False
        
    @property
    def end(self):
        """Gets the end point, as an index in the grid."""
        return self._end
    
    @end.setter
    def end(self, point):
        """
        Sets the end point.
        
        Args::
        
            point (iterable of int-like or float-like): 3-element iterable
                that contains the ending point. If this is int-like, i.e.
                isinstance(point, (int, long, np.int0, np.int8, ...)) is
                True, then the point is taken as an index, otherwise as
                a point in cartesian space.
        """
        self._end = self._smart_point(point, type="auto")
        self._upToDate = False
        
    @property
    def start(self):
        """Gets the starting point, as an index in the grid."""
        return self._start
    
    @start.setter
    def start(self, point):
        """
        Sets the start point.
        
        Args::
        
            point (iterable of int-like or float-like): 3-element iterable
                that contains the starting point. If this is int-like, i.e.
                isinstance(point, (int, long, np.int0, np.int8, ...)) is
                True, then the point is taken as an index, otherwise as
                a point in cartesian space.
        """
        self._start = self._smart_point(point, type="auto")
        self._upToDate = False

    #@autojit
    def eval(self):
        """
        Evaluate to find the lowest cost path.
        
        Args::
        
            None
                
        Returns::
        
            True if the path was completed successfully. False otherwise
        """
        # ensure all parameters have been set
        if self.grid is None:
            return False
        if self._end is None:
            return False
        if self._start is None:
            return False
        # For N = the stepsize in connections, and N != 1, but
        # end - start != iN, with i an integer, then no direct path to the
        # end node exists, though it could wrap periodically to reach the end.
        # Evaluation will take a long time, until all accessible boxes
        # have been checked, but that is unavoidable.
        #
        # if there is a problem with these closures (nested functions)
        # something like "ERROR (variable) must have a single type" or
        # "(var) is a cell variable", then see
        # http://numba.pydata.org/numba-doc/dev/pythonstuff.html
        #@jit('b1(int_[:], int_[:])')
        def path_cmp(i, j):
            """Comparable to use when comparing path values."""
            # path_value must be set in the parent namespace before this
            # function is called.
            i = ijk_to_i(i)
            j = ijk_to_i(j)
            return path_value[i] > path_value[j]
        
        #@jit('int_[:,:](int_[:])')
        def get_adjacent(center):
            """
            --------
            Synopsis
            --------
            Returns a tuple of indices for adjacent nodes that have yet to be
            considered central nodes.
    
            ----------
            Parameters
            ----------
            :center (tuple of ints): Index of the central node whose
                adjacent nodes are sought.
            :returns (tuple of tuple of ints): Indices of adjacent nodes that
                have not yet been central nodes.
            """
            dim = np.array(self.grid.shape, dtype=int)
            periodic = self.grid.is_periodic
            adjacent = []
            ijk = np.zeros_like(center)
            for dijk in self.connections:
                # axis-aligned indices for the prospective adjacent node
                ijk = center + dijk
                # adjust only those directions for which periodicity, set
                # by the grid itself, is True. This is the function:
                #
                #   ijk = ijk - n*(i // n)
                #
                # applied only to those directions that are periodic
                ijk[periodic] = ijk[periodic] - \
                                dim[periodic]*(ijk[periodic] // dim[periodic])
                # periodic searching was handled immediately above,
                # t.f. exclude potential nodes that lie beyond the boundaries
                # of the grid
                if np.all(0 <= ijk) and np.all(ijk < dim):
                    # axis-aligned to flattened index
                    i = ijk_to_i(ijk)
                    # Exclude node if it has already been a center.
                    # path_set must be defined in the parent namespace.
                    if not path_set[i]:
                        adjacent.append(tuple(ijk))
            return np.array(adjacent)
        # end 'def get_adjacent(center_node_index):'
        
        #@jit('int_(int_[:])')
        def ijk_to_i(ijk):
            """
            Returns the flattened index *i* corresponding to the axis-aligned
            index *ijk*, i.e. self.grid.flat[i] === self.grid[ijk]
            """
            dim = self.grid.shape
            stride = np.array([dim[-1]*dim[-2], dim[-1], 1]) # C-style
            #stride = np.array([1, dim[0], dim[0]*dim[1]]) # F-style
            return np.sum(stride*ijk)
    
        # shortest path to each node, None if node not yet in path
        # *NOTE* this variable is used in get_adjacent, so
        # though it is declared here, it is referenced, though not
        # used, above.
        path_value = self.grid.size*[None]
        # index to the previous node in the shortest path to each node
        prev_node = self.grid.size*[None]
        # bitarray holds those nodes that have (1) or have not (0) been
        # considered as central nodes, i.e. nodes whose path has been
        # established. *NOTE* this variable is used in get_adjacent, so
        # though it is declared here, it is referenced, though not
        # used, above.
        path_set = bitarray(self.grid.size)
        try:
            path_set.setall(0)                          # pylint: disable=E1101
        except AttributeError:
            for i in xrange(len(path_set)):
                path_set[i] = False
        # queue of node indices to serve as the central node, these
        # are the adjacent nodes with lowest value from previous iterations
        queue = [self.start]
        # keep vector sorted in descending order relative to the value
        # stored in path_value, so that the last has the shortest path
    
        # ----------- FORWARD ----------- #
        central = queue.pop()
        icentral = ijk_to_i(central)
        end = self.end
        prev_node[icentral] = None # reverse linked list
        path_value[icentral] = 0.0
        path_set[icentral] = 1
        while np.any(central != end):
            for adjacent in get_adjacent(central):
                iadjacent = ijk_to_i(adjacent)
                # calc function to get to adjacent from central
                path = self.cost_function(central, adjacent)
                if path < 0:
                    raise ValueError("A negative value for the cost function "
                                     "was encountered, but it must be "
                                     "everywhere positive. May I suggest "
                                     "f(i,j) --> exp(f(i,j)), or similar?")
                path += path_value[icentral]
                # has a path to adjacent already been found?
                if path_value[iadjacent] is not None:
                    # is the new path lower cost?
                    if path < path_value[iadjacent]:
                        # remove adjacent node from previous location
                        # in queue. Why not check for 0? Because
                        # if we get here, adjacent will already be in
                        # queue, and bisect, which returns the
                        # right insertion point, will return a value in
                        # the range [1,len(queue)]
                        i = bisect(queue,
                                   adjacent,
                                   comparable=path_cmp)-1
                        adjacent = queue.pop(i)
                        iadjacent = ijk_to_i(adjacent)
                    # no, then move to the next adjacent node
                    else:
                        continue
                # (re)add adjacent to the queue
                prev_node[iadjacent] = central
                path_value[iadjacent] = path
                insort(queue,
                       adjacent,
                       comparable=path_cmp)
            # get the next central node, i.e. the adjacent node with the
            # cheapest path, as there is no cheaper way to get to this
            # soon-to-be-central node.
            central = queue.pop()
            icentral = ijk_to_i(central)
            path_set[icentral] = 1
        # ----------- BACKWARD ----------- #
        pathway = []
        cost = [] # cost at each point
        total_cost = path_value[ijk_to_i(central)]
        while central is not None:
            pathway += [central]
            cost += path_value[ijk_to_i(central)] # incremental cost
            central = prev_node[central]
        pathway.reverse()
        cost.reverse()
        # x = (cumx1, cumx2, cumx3, ..., cumxN) - (0, cumx1, cumx2, ...)
        cost = tuple(np.array(cost) - np.array([0.0] + cost[:-1]))
        # save the path and cost
        self._path = tuple(pathway)
        self._cost = (cost, total_cost)
        self._upToDate = True
        # done
        return True
#end class DijkstraPath
