"""
...
"""

# external
import os
import numpy as np
import psutil
from numbers import Integral
from filelock import FileLock

# aimmd imports
from .base import AbstractCache
from ..core.utils import extract_folder_and_name, extend_array

# auxiliray functions
def save_npy(fname, array):
    """Safe save"""
    folder, name = extract_folder_and_name(fname)
    temp = f'{folder}/.{name}'
    lock = f'{folder}/.{name}.lock'
    with FileLock(lock):
        np.save(temp, array)
        os.replace(temp, fname)

def load_npy(fname, timeout=5.):
    """Safe load"""
    if not os.path.exists(fname):
        return None
    try:
        folder, name = extract_folder_and_name(fname)
        lock = f'{folder}/.{name}.lock'
        with FileLock(lock, timeout=timeout):
            return np.load(fname)
    except:
        return None

def update_npy(fname, data, indices):
    """Please document very cool function
    Works also with indices integral, then data will be added another d."""
    
    # process and get info
    if isinstance(indices, Integral):
        data = [data]
    data = np.atleast_1d(data)
    indices = np.asarray(indices).flatten()
    min_size = int(indices.max()) + 1
    data_shape = data.shape
    data_dtype = data.dtype
    data_descr = data_dtype.descr
    if len(data_descr) > 1:
        raise RuntimeError(f'only simple arrays allowed')
    
    # create
    if not os.path.exists(fname):
        new_shape = (min_size,) + data_shape[1:]
        result = np.zeros(new_shape, dtype=data_dtype)
        result[indices] = data
        save_npy(fname, result)
    
    # update in place
    # get row size
    rowsize = data.itemsize
    if len(data_shape) >= 1:
        rowsize *= np.prod(data_shape[1:])
    rowsize = int(rowsize)
    
    # go thrugh file
    folder, name = extract_folder_and_name(fname)
    with FileLock(f'{folder}/.{name}.lock'):
        with open(fname, "r+b") as file:
            header = file.read(128)
            
            # check descr
            descr_begin = header.find(b"'descr': ") + 9
            descr_end = header.find(b", 'fortran")
            descr = header[descr_begin:descr_end]
            data_descr = f"'{data_descr[0][1]}'".encode()
            if descr != data_descr:
                descr = descr.decode('latin1')
                data_descr = data_descr.decode('latin1')
                raise RuntimeError(f'compute result must have '
                   f'descr {descr}, got {str(data_descr)} '
                   f'instead; consider deleting {fname!r} first')
            
            # get shape
            shape_begin = header.find(b"'shape': (") + 9
            shape_end = shape_begin + header[shape_begin:].find(b'),') + 1
            shape = header[shape_begin + 1:shape_end - 1].decode('latin1')
            shape = tuple([int(s) for s in shape.split(',') if s.strip()])
            
            if shape[1:] != data_shape[1:]:
                raise RuntimeError(f'compute result must have '
                   f'shape {(-1, ) + shape[1:]}, got {data_shape} '
                   f'instead; consider deleting {fname!r} first')
            
            # update shape to final size
            new_size = max(int(min_size), int(shape[0]))
            
            # resize
            if shape[0] != new_size:
                file.truncate(128 + new_size * rowsize)
            
            # write rows frame by frame
            for i, rowdata in zip(indices, data):
                file.seek(128 + i * rowsize)
                file.write(rowdata.tobytes())
            
            # write header for last (more robust with np.load)
            if shape[0] != new_size:
                new_shape = (new_size,) + shape[1:]
                header = (header[:shape_begin] +
                          str(new_shape).encode('latin1') +
                          header[shape_end:])[:127] + b"\n"
                file.seek(0)
                file.write(header)
            
            file.flush()
            os.fsync(file.fileno())


# array cache
class NpyReaderCache(AbstractCache):
    
    max_size = int(psutil.virtual_memory().available)
    
    def _open(self, fname):
        result = load_npy(fname)
        if result is not None:
            result.flags.writeable = False
        return result
    
    def _extend(self, instance, min_length):
        return extend(instance, min_length)
