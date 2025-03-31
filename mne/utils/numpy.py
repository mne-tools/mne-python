# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import io
import os

import numpy

AnyFile = str | bytes | os.PathLike | io.IOBase


def numpy_fromfile(
    file: AnyFile, dtype: numpy.typing.DTypeLike = float, count: int = -1
):
    """numpy.fromfile() wrapper, handling io.BytesIO file-like streams.

    Numpy requires open files to be actual files on disk, i.e., must support
    file.fileno(), so it fails with file-like streams such as io.BytesIO().

    If numpy.fromfile() fails due to no file.fileno() support, this wrapper
    reads the required bytes from file and redirects the call to
    numpy.frombuffer().

    See https://github.com/numpy/numpy/issues/2230
    """
    try:
        return numpy.fromfile(file, dtype=dtype, count=count)
    except io.UnsupportedOperation as e:
        if not (e.args and e.args[0] == "fileno" and isinstance(file, io.IOBase)):
            raise  # Nothing I can do about it
        dtype = numpy.dtype(dtype)
        buffer = file.read(dtype.itemsize * count)
        return numpy.frombuffer(buffer, dtype=dtype, count=count)
