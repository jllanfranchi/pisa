# pylint: disable=redefined-outer-name


"""
Collection of useful vectorized functions
"""

from __future__ import absolute_import, print_function

__version__ = '0.1'
__author__ = 'Philipp Eller (pde3@psu.edu)'

import math

import numpy as np
from numba import guvectorize

from pisa import FTYPE, TARGET
from pisa.utils.numba_tools import WHERE


FX = 'f4' if FTYPE == np.float32 else 'f8'


def imultiply_and_scale(scale, values, out):
    """Multiply and scale augmented assignment .. ::

        out[:] *= scale * values[:]

    Parameters
    ----------
    scale : scalar
    values : SmartArray
    out : SmartArray

    """
    imultiply_and_scale_gufunc(scale, values.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def scale(scale, values, out):
    """Scale .. ::

        out[:] = scale * values[:]

    Parameters
    ----------
    scale : scalar
    values : array
    out : array

    """
    scale_gufunc(scale, values.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def imultiply(values, out):
    """Multiply augmented assignment of two arrays .. ::

        out[:] *= values[:]

    Parameters
    ----------
    values : SmartArray
    out : SmartArray

    """
    imultiply_gufunc(values.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def idivide(values, out):
    """Divide augmented assignment .. ::

        out[:] /= values[:]

    Division by zero results in 0 for that element.
    """
    idivide_gufunc(values.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def set(values, out):  # pylint: disable=redefined-builtin
    """Set array values from another array .. ::

        out[:] = values[:]

    """
    set_gufunc(values.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def square(values, out):
    """Square values .. ::

        out[:] = values[:]**2

    """
    square_gufunc(values.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def sqrt(values, out):
    """Square root of values .. ::

        out[:] = sqrt(values[:])

    """
    sqrt_gufunc(values.get(WHERE), out=out.get(WHERE))
    out.mark_changed(WHERE)


def replace(counts, min_count, values, out):
    """Replace `out[i]` with `values[i]` when `counts[i]` > `min_count`"""
    replace_gufunc(
        counts.get(WHERE),
        min_count,
        values.get(WHERE),
        out=out.get(WHERE),
    )


@guvectorize([f'({FX}, {FX}[:], {FX}[:])'], '(),()->()', target=TARGET)
def imultiply_and_scale_gufunc(scale, values, out):
    """Augmented assigment multiply and scale .. ::

        out[:] *= scale * values[:]

    """
    out[0] *= scale * values[0]


@guvectorize([f'({FX}, {FX}[:], {FX}[:])'], '(),()->()', target=TARGET)
def scale_gufunc(scale, values, out):
    """Scale .. ::

        out[:] = scale * values[:]

    """
    out[0] = scale[0] * values[0]


@guvectorize([f'({FX}[:], {FX}[:])'], '()->()', target=TARGET)
def imultiply_gufunc(values, out):
    """Multipy augmented assignment .. ::

        out[:] *= values[:]

    """
    out[0] *= values[0]


@guvectorize([f'({FX}[:], {FX}[:])'], '()->()', target=TARGET)
def idivide_gufunc(values, out):
    """Divide augmented assignment .. ::

        out[:] /= values[:]

    Division by zero results in 0 for that element.
    """
    if values[0] == 0.:
        out[0] = 0.
    else:
        out[0] /= values[0]


@guvectorize([f'({FX}[:], {FX}[:])'], '()->()', target=TARGET)
def set_gufunc(values, out):
    """Set array values from another array .. ::

        out[:] = values[:]

    """
    out[0] = values[0]


@guvectorize([f'({FX}[:], {FX}[:])'], '()->()', target=TARGET)
def square_gufunc(values, out):
    """Square values .. ::

        out[:] = values[:]**2

    """
    out[0] = values[0]**2


@guvectorize([f'({FX}[:], {FX}[:])'], '()->()', target=TARGET)
def sqrt_gufunc(values, out):
    """Square root of values .. ::

        out[:] = sqrt(values[:])

    """
    out[0] = math.sqrt(values[0])


@guvectorize([f'({FX}[:], i4[:], {FX}[:], {FX}[:])'], '(),(),()->()', target=TARGET)
def replace_gufunc(counts, min_count, values, out):
    """Replace `out[i]` with `values[i]` when `counts[i]` > `min_count`"""
    if counts[0] > min_count[0]:
        out[0] = values[0]


def test_imultiply_and_scale():
    """Unit tests for function ``imultiply_and_scale``"""
    from numba import SmartArray
    a = np.linspace(0, 1, 1000, dtype=FTYPE)
    a = SmartArray(a)

    out = np.ones_like(a)
    out = SmartArray(out)

    imultiply_and_scale(10., a, out)

    assert np.allclose(out.get('host'), np.linspace(0, 10, 1000, dtype=FTYPE))


if __name__ == '__main__':
    test_imultiply_and_scale()
