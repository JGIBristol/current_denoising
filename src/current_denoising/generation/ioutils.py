"""
Input and output utilities
"""

import pathlib

import numpy as np


class IOError(Exception):
    """General error with I/O"""


def read_currents(path: pathlib.Path) -> np.ndarray:
    """
    Read a .dat file holding current data and return a 720x1440 shaped numpy array giving
    the current in m/s (I think)

    :param path: location of the .dat file; current data is located in
                 data/projects/SING/richard_stuff/Table2/currents/ on the RDSF
    :returns: a numpy array holding current speed
    :raises ValueE
    """
    dtype = np.dtype("<f4")
    shape = 720, 1440

    type_size = np.dtype(dtype).itemsize

    with open(path, "rb") as f:
        first_record_len = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=dtype, count=first_record_len // type_size)

        closing_record_len = np.fromfile(f, dtype=np.int32, count=1)[0]
        if closing_record_len != first_record_len:
            raise IOError("Close marker does not match the opener.")

    return data.reshape(shape)
