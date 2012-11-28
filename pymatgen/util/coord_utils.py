#!/usr/bin/env python

"""
Utilities for manipulating coordinates or list of coordinates
"""

from __future__ import division

__author__ = "Shyue Ping Ong"
__copyright__ = "Copyright 2011, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyue@mit.edu"
__date__ = "Nov 27, 2011"

import numpy as np
import math


def in_coord_list(coord_list, coord, **kwargs):
    """
    Tests if a particular coord is within a coord_list using numpy.allclose.

    Args:
        coord_list:
            List of coords to test
        coord:
            Specific coordinates
        kwargs:
            keyword arguments supported by numpy.allclose. Please refer to
            documentation for numpy.allclose.
    """
    for test in coord_list:
        if np.allclose(test, coord, **kwargs):
            return True
    return False


def get_linear_interpolated_value(x_values, y_values, x):
    """
    Returns an interpolated value by linear interpolation between two values.
    This method is written to avoid dependency on scipy, which causes issues on
    threading servers.

    Args:
        x_values:
            Sequence of x values.
        y_values:
            Corresponding sequence of y values
        x:
            Get value at particular x
    """
    val_dict = dict(zip(x_values, y_values))
    if x < min(x_values) or x > max(x_values):
        raise ValueError("x is out of range of provided x_values")

    for val in sorted(x_values):
        if val < x:
            x1 = val
        else:
            x2 = val
            break

    y1 = val_dict[x1]
    y2 = val_dict[x2]

    return y1 + (y2 - y1) / (x2 - x1) * (x - x1)


def coords_to_unit_cell(coords):
    """
    Map a set of fractional coordinates into the unit cell, i.e.
    0 <= a < 1.

    Args:
        coords:
            A sequence of fractional coords.

    Returns:
        The coords mapped to within the unit cell.
    """
    return [i - math.floor(i) for i in coords]


def pbc_diff(fcoord1, fcoord2):
    """
    Returns the 'fractional distance' between two coordinates taking into
    account periodic boundary conditions.

    Args:
        fcoord1:
            First fractional coordinate.
        fcoord2:
            Second fractional coordinate.

    Returns:
        Fractional distance. Each coordinate must have the property that
        abs(a) < 0.5. Examples:
        pbc_diff([0.1, 0.1, 0.1], [0.3, 0.5, 0.9]) = [-0.2, -0.4, 0.2]
        pbc_diff([0.9, 0.1, 1.01], [0.3, 0.5, 0.9]) = [-0.4, -0.4, 0.11]
    """
    fdist = np.array(coords_to_unit_cell(fcoord1)) - \
            np.array(coords_to_unit_cell(fcoord2))
    return [a if abs(a) < 0.5 else a - a/abs(a) for a in fdist]


def in_coord_list_pbc(coord_list, coord, **kwargs):
    """
    Tests if a particular coord is within a coord_list using numpy.allclose.

    Args:
        coord_list:
            List of coords to test
        coord:
            Specific coordinates
        kwargs:
            keyword arguments supported by numpy.allclose. Please refer to
            documentation for numpy.allclose.
    """
    for test in coord_list:
        fdiff = pbc_diff(test, coord)
        if np.allclose(fdiff, [0,0,0], **kwargs):
            return True
    return False