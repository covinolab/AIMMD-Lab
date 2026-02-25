"""
aimmd.path.chainreader
======================

Sequential reader abstraction over multiple trajectory readers.

This module defines :class:`ChainReader`, a lightweight wrapper that presents
multiple reader-like objects as a single continuous trajectory.

Typical use case
----------------
When a Path spans multiple trajectory files (forward/backward segments or
multi-file continuations), `ChainReader` allows them to be accessed as if
they were a single reader:

    >>> reader = ChainReader(reader1, reader2)
    >>> for ts in reader:
    ...     ...

Design goals
------------
- No data copying.
- Preserve underlying reader indexing.
- Support slicing to create sub-readers.
- Keep iteration cheap and Python-level only.

Notes
-----
- Readers must support `__len__` and `__getitem__`.
- `ChainReader[i]` performs local index resolution.
- `__iter__` resets automatically after exhaustion.
"""

# external
import numpy as np
import bisect
from numbers import Integral


class ChainReader:
    """
    Iterator and indexable view over multiple readers.

    Parameters
    ----------
    *readers : iterable
        Reader-like objects (e.g., MDAnalysis readers) supporting
        `__len__` and `__getitem__`.

    Notes
    -----
    - The total length is the sum of the individual lengths.
    - Random access is resolved via cumulative offsets.
    """

    def __init__(self, *readers):
        self.readers = readers
        self._keys = iter(range(self.lengths.sum()))
        self._length = self.lengths.sum()

    @property
    def lengths(self):
        """
        Per-reader frame counts.

        Returns
        -------
        numpy.ndarray
            1D array containing the length of each reader.
        """
        return np.array([len(reader) for reader in self.readers], dtype=int)

    @property
    def n_atoms(self):
        """
        Number of atoms per frame.

        Returns
        -------
        int or None
            Number of atoms from the first reader, if any.
        """
        if self.readers:
            return self.readers[0].trajectory.n_atoms

    def __len__(self):
        """Total number of frames across all readers."""
        return self._length

    def __iter__(self):
        """
        Return iterator over all frames.

        Notes
        -----
        Iteration state is reset automatically after exhaustion.
        """
        return self

    def __next__(self):
        """
        Iterate sequentially over all frames.

        Raises
        ------
        StopIteration
            When all frames have been consumed.
        """
        try:
            i = next(self._keys)
            k, i = self._get_local_index(i)
            return self.readers[k][i]
        except StopIteration:
            # Reset iterator for possible reuse
            self._keys = iter(range(self.lengths.sum()))
            self._length = self.lengths.sum()
            raise StopIteration

    @property
    def offsets(self):
        """
        Cumulative frame offsets per reader.

        Returns
        -------
        numpy.ndarray
            Cumulative sum of lengths, used for index resolution.
        """
        return np.cumsum(self.lengths)

    def _get_local_index(self, i):
        """
        Map global index to (reader_index, local_index).

        Parameters
        ----------
        i : int
            Global frame index in range [0, total_length).

        Returns
        -------
        tuple[int, int]
            (reader_index, local_index)

        Raises
        ------
        IndexError
            If index is outside valid range.
        """
        offsets = self.offsets
        k = bisect.bisect_right(offsets, i)
        if k >= len(offsets):
            raise IndexError(i)
        if k:
            return k, i - offsets[k - 1]
        return k, i

    def __getitem__(self, key):
        """
        Random access or slicing.

        Parameters
        ----------
        key : int or slice
            Frame index or slicing object.

        Returns
        -------
        object or ChainReader
            Frame for integer indexing,
            new ChainReader for slicing.
        """
        if isinstance(key, Integral):
            k, i = self._get_local_index(key)
            return self.readers[k][i]

        # Return sliced ChainReader
        result = object.__new__(ChainReader)
        result.readers = self.readers
        keys = np.arange(len(self))[key]
        result._keys = iter(keys)
        result._length = len(keys)
        return result
