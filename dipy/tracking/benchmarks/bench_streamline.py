""" Benchmarks for functions related to streamline

Run all benchmarks with::

    import dipy.tracking as dipytracking
    dipytracking.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also run
the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_streamline.py
"""
import numpy as np

from dipy.tracking.streamline import set_number_of_points, length
from dipy.tracking.tests.test_streamline import set_number_of_points_python, length_python

from numpy.testing import measure


def bench_resample():
    repeat = 1000
    nb_points = 42
    streamline = np.random.rand(1000, 3)

    print("Timing set_number_of_points() in Cython")
    cython_time = measure("set_number_of_points(streamline, nb_points)", repeat)
    print("Cython time: {0:.2}sec".format(cython_time))

    python_time = measure("set_number_of_points_python(streamline, nb_points)", repeat)
    print("Python time: {0:.2}sec".format(python_time))
    print("Speed up of {0}x".format(python_time/cython_time))


def bench_length():
    repeat = 1000
    streamline = np.random.rand(1000, 3)

    print("Timing length() in Cython")
    cython_time = measure("length(streamline)", repeat)
    print("Cython time: {0:.2}sec".format(cython_time))

    python_time = measure("length_python(streamline)", repeat)
    print("Python time: {0:.2}sec".format(python_time))
    print("Speed up of {0}x".format(python_time/cython_time))
