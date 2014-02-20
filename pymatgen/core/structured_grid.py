#!/usr/bin/env python

"""
This module defines the classes relating to 3D lattices.
"""

from __future__ import division

__author__ = "Branden Kappes"
__copyright__ = "Copyright 2014, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Branden Kappes"
__email__ = "branden.kappes@gmail.com"
__status__ = "Developmental"
__date__ = "Jan 8, 2014"

import itertools
import numpy as np
from pymatgen.core.structure import Structure

class StructuredGrid(np.ndarray):
    """
    A volume mapping object. Essentially, this is a matrix that
    maps a scalar (default), vector, or general tensor field to a regular
    grid that spans the spatial domain defined by the Structure object
    contained in the StructuredGrid.
    """
    
    # number of dimensions in the structured grids. 
    __NDIM = 3
    
    def __new__(subclass, structure, shape,
                dtype=float,
                buffer=None):
        """
        Called upon explicit construction of a StructuredGrid object --
        as opposed to view casting or new-from-template.
        
        Args::
        
            structure (required): Structure object that defines
                the spatial range spanned by this grid.
        
            shape (required): The number of volume elements
                (voxels) along each spatial direction, i.e.
                (Nx, Ny, Nz).
        
            dtype (optional): The type of data stored in this grid
                Default: float. Other examples::
                
                    multiple, named scalar fields:
                        dtype=np.dtype([('A', float), ..., ('N', float)])
                    
                    N-dimensional vector field:
                        dtype=np.dtype([(float, (N,))])
                        
                    MxN subarray:
                        dtype=np.dtype([(float, (M, N))])
                        
            buffer (optional): Object exposing the buffer interface.
                Used to fill the array with data.
        
        Returns::
            New StructuredGrid instance.
        """
        if len(shape) != StructuredGrid.__NDIM:
            raise ValueError("Structured grid must be three dimensional, "
                             "i.e. (Nx, Ny, Nz)")
        obj = np.ndarray.__new__(subclass, shape, dtype=dtype, buffer=buffer,
                                 offset=0, strides=None, order='C')
        # set the structure that defines the chemical and spatial extents of
        # the grid
        obj.structure = structure
        obj._is_periodic = np.array([True, True, True])
        return obj
    
    def __array_finalize__(self, obj):
        """
        Mechanism used by numpy.ndarray to handle instantiation of objects
        constructed either explicitly or through view casting or new-from-
        template.
        """
        # We could be here in 3 ways:
        # From an explicit constructor, e.g. StructuredGrid(...):
        #     obj is None in that case
        if obj is None:
            return
        # From either view casting, e.g. arr.view(StructuredGrid):
        #     obj is arr
        #     type(obj) may, or may not, be StructuredGrid
        # or new-from-template, e.g. grid[:3]:
        #     type(obj) is StructuredGrid
        # In either of these last two cases, StructuredGrid.__new__ will not
        # be called, and so it is here that the additional attributes of
        # the StructuredGrid are set.
        #
        # Get the corresponding Structure object, should it exist, or create
        # an empty Structure
        self.structure = getattr(obj, 'structure',
                                 Structure(lattice=np.eye(StructuredGrid.__NDIM),
                                           species=[],
                                           coords=[]))
        obj._is_periodic = getattr(obj, '_is_periodic',
                                   np.array([True, True, True]))
    
    def __iter__(self):
        """
        Iterates through every element in the grid. To simplify access
        operations, this is set to C-continuous (row dominant) order, That is,
        the last dimension (z) moves the fastest, then y, then finally, x,
        i.e.::
        
            (0, 0, 0)
            (0, 0, 1)
            ...
            (0, 0, Nz-1)
            (0, 1, 0)
            (0, 1, 1)
            ...
            (0, Ny-1, Nz-1)
            (1, 0, 0)
            (1, 0, 1)
            ...
            (Nx-1, Ny-1, Nz-2)
            (Nx-1, Ny-1, Nz-1)
        
        Returns::
            *yield*s the next element in the grid, or raises StopIteration
            after the last value has been yielded.
        """
        for x in self.flat:
            yield x
    
    def _roundup_index(self, ijk):
        """
        Where possible, i.e. for fully qualified voxel indicies, ensures
        that each index lies between [0, Np) for p = x, y, z, respectively
        for all integer (not slice, not ellipsis) indices. The reason for
        this limitation is purely practical: I can't think of any good way
        to handle slices, i.e.
        
        .. code:: python
        
            rval = grid[i, j, k]
            print rval
        
        would call :code:`grid.__getitem__` exactly once. Easy. However,
        
        .. code:: python
        
            # EITHER...
            rval = grid[i, j, :]
            # OR...
            rval = grid[i, j, ...]
            # OR...
            rval = grid[i, j]
            # THEN
            print rval
        
        are all three practically equivalent and will call
        :code:`rval.__getitem__(k)` a total of *Nz* times, with
        k = -Nz+1, -Nz+2, ..., -1 on a 1-D StructuredGrid object,
        that is, self.ndim != StructuredGrid.__NDIM. The question,
        then, is along what direction is this vector? In the example above it
        is z. But :code:`rval.__getitem__(q)`, q = i, j, or k, would
        similarly be called *Np* times, p = x, y, or z. Because I would have
        lost information about which direction, or directions, remain,
        I don't know whether this remaining direction is periodic.
        
        For this reason, only fully specified voxel indices are periodic-aware.
        """
        # PA: This line/section is periodic aware
        # NPA: This line/section is not periodic aware
        try:
            # will raise a TypeError exception if idx does not have a length,
            # i.e. if idx is an int, a slice, or an ellipsis => NPA
            if len(ijk) == StructuredGrid.__NDIM:
                dim = self.shape
                newIndex = list(ijk)
                for i, obj in enumerate(ijk):
                    # NPA: one index is a slice or ellipsis
                    if obj.__class__ is slice or obj is Ellipsis:
                        continue
                    # PA: this index is an int
                    if self._is_periodic[i]:
                        newIndex[i] %= dim[i]
                    # PA: periodically aware, but not enforced per user request
                    elif newIndex[i] < 0 or newIndex[i] >= dim[i]:
                        raise IndexError("Index %d is out of bounds for "
                                         "axis %d with size %d" % \
                                         (newIndex[i], i, dim[i]))
                ijk = tuple(newIndex)
        except TypeError: # NPA: idx is an int, slice, or ellipsis
            pass
        # done, return the resulting index
        return ijk
    
    def __getitem__(self, idx):
        """
        Overload the access operator, [...], to be aware of periodicity. *NOTE*
        This changes the default behavior in that negative indices are only
        allowed when periodicity is enforced or if slices are taken.
        """
        ijk = self._roundup_index(idx)
        return np.ndarray.__getitem__(self, ijk)
    
    def __setitem__(self, idx, val):
        """
        Sets the value at idx, considering periodicity. *NOTE*
        This changes the default behavior in that negative indices are only
        allowed when periodicity is enforced or if slices are taken.
        """
        ijk = self._roundup_index(idx)
        np.ndarray.__setitem__(self, ijk, val)
        
    @property
    def is_periodic(self):
        """
        Returns whether (True) or not (False) indexing the StructuredGrid
        object should be done periodically.
        
        .. Returns::
        
            numpy.ndarray((3,), dtype=bool)
        """
        # ensure the user cannot inadvertently invalidate _is_periodic
        # by gaining access to the ndarray buffer.
        return np.copy(self._is_periodic)
    
    @is_periodic.setter
    def is_periodic(self, periodic):
        """
        Sets whether indexing the StructuredGrid should be done periodically.
        
        .. attributes::
        
            periodic (bool or 3-element iterable of bool, required):
                True - index periodically. False - Do not index periodically.
        """
        try:
            # periodic is an iterable
            if len(periodic) == StructuredGrid.__NDIM:
                self._is_periodic = np.array(periodic, dtype=bool, copy=True)
            else:
                raise ValueError("The periodicity of the StructuredGrid "
                                 "must match its dimensionality.")
        except TypeError:
            # periodic is a scalar
            self._is_periodic = np.array(StructuredGrid.__NDIM*[periodic],
                                         dtype=bool)

    def at(self, coord):
        """
        Returns the value from the grid voxel that contains the coordinate
        coord = p(x, y, z). This is equivalent to the much more convoluted
        grid[grid.index((x, y, z))].
        
        Args::
        
            coord (required, 3-element iterable of floats): coordinate
            
            periodic (optional, bool): Whether to search periodic images of
                the structure. Default: True
        
        Returns::
        
            Object stored at voxel (i, j, k) containing p(x, y, z)
        """
        return self[self.index(coord)]
            
    def index(self, coord):
        """
        Returns the index tuple v(i, j, k) of the voxel that contains
        the cordinate coord = p(x, y, z). *v* ``contains`` p if, in
        fractional coordinates fp(fx, fy, fz), :math:`i/Nx <= fx < (i+1)/Nx`
        and :math:`j/Ny <= fy < (j+1)/Ny` and :math:`k/Nz <= fz < (k+1)/Nz`.
        
        Args::
        
            coord (required, 3-element iterable of floats): coordinate
        
        Returns::
            
            Index tuple of the voxel (i, j, k) that contains (x, y, z).
            This index may lie outside the boundaries of the StructuredGrid.
        """
        assert len(coord) == StructuredGrid.__NDIM
        # coord in fractional coordinates
        frac = np.dot(coord, self.structure.lattice.inv_matrix)
        # frac as (i, j, k), see the description of index 
        frac = np.floor(frac * self.shape).astype(int)
        return tuple(frac)
#end class StructuredGrid
