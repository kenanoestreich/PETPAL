"""A collection of functions to compute TACs as explicit convolutions for common Tissue Compartment Models (TCMs).


TODO:
    * Add the derivations of the solutions to the TCMs in the module docstring.
"""

import numba
import numpy as np


def calc_convolution_with_check(f: np.ndarray[float], g: np.ndarray[float], dt: float) -> np.ndarray:
    """Performs a discrete convolution of two arrays, assumed to represent time-series data.
    
    Let ``f``:math:`=f(t)` and ``g``:math:`=g(t)` where both functions are 0 for :math:`t\leq0`. Then,
    the output, :math:`h(t)`, is
    .. math::
        h(t) = \int_{0}^{t}f(s)g(s-t)\mathrm{d}s
    
    Args:
        f (np.ndarray[float]): Array containing the values for the input function.
        g (np.ndarray[float]): Array containing values for the response function.
        dt (np.ndarray[float]): The step-size, in the time-domain, between samples for ``f`` and ``g``.

    Returns:
        (np.ndarray): Convolution of the two arrays scaled by ``dt``.
        
    Notes:
        This function does not use `numba.njit()`.
    """
    assert len(f) == len(g), f"The provided arrays must have the same lengths! f:{len(f):<6} and g:{len(g):<6}."
    vals = np.convolve(f, g, mode='full')
    return vals[:len(f)] * dt

@numba.njit()
def response_function_1tcm_c1(t: np.ndarray[float], k1: float, k2: float) -> np.ndarray:
    """The response function for the 1TCM :math:`f(t)=k_1 e^{-k_{2}t}`
    
    Args:
        t (np.ndarray[float]): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from tissue compartment back to plasma/blood.

    Returns:
        (np.ndarray[float]): Array containing response function values given the constants.
    """
    return k1 * np.exp(-k2 * t)


@numba.njit()
def response_function_2tcm_with_k4zero_c1(t: np.ndarray[float], k1: float, k2: float, k3: float) -> np.ndarray:
    """The response function for first compartment in the 2TCM with :math:`k_{4}=0`; :math:`f(t)=k_{1}e^{-(k_{2} + k_{3})t}`.
    
    Args:
        t (np.ndarray[float]): Array containing time-points where :math:`t\geq0`.
        k1: Rate constant for transport from plasma/blood to tissue compartment.
        k2: Rate constant for transport from tissue compartment back to plasma/blood.
        k3: Rate constant for transport from tissue compartment to irreversible compartment.

    Returns:
        (np.ndarray[float]): Array containing response function values for first compartment given the constants.
    """
    return k1 * np.exp(-(k2 + k3) * t)
