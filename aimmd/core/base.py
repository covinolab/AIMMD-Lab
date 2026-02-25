"""
aimmd.core.base
===============

Foundational abstract/base utilities used throughout AIMMD.

This module defines :class:`AbstractArray`, a lightweight proxy that enables
objects to behave like NumPy arrays *without* inheriting from ``numpy.ndarray``.

Why this exists in AIMMD
------------------------
AIMMD frequently exposes *derived* or *backed* data as array-like views. For
example, the package defines properties such as ``weights``, ``accepted``,
``exclude_from``, and ``shooting_indices`` that look like arrays but actually
map to attributes of underlying path objects. In that usage pattern
(:class:`~aimmd.pathensemble._properties.PathProperties`), the array-like object:

- supports scalar indexing (``obj[i]``) by mapping to one underlying element,
- supports slicing / fancy indexing (``obj[:]`` / ``obj[idx]``) by returning a
  NumPy array materialized from underlying objects,
- supports assignment (``obj[:] = values``) by *writing back* into underlying
  objects (attribute updates).

This matches the intended contract of :class:`AbstractArray`: it provides a
common array façade while allowing subclasses to control where data come from
and where writes go. :contentReference[oaicite:3]{index=3}

Key idea
--------
Subclasses implement::

    def _array(self):
        ...

which must return an array-like object representing the current contents.
By default, :class:`AbstractArray` forwards common protocols to that backing
array.

**However, subclasses may (and in AIMMD often do) override `__getitem__`
and/or implement `__setitem__`** to support efficient or write-through access.
In particular, :class:`~aimmd.pathensemble._properties.PathProperties` overrides
``__getitem__`` to avoid materializing the full array when only a single
element is requested. :contentReference[oaicite:4]{index=4}

Subclass contract (as used in AIMMD)
------------------------------------
Required:
- ``_array(self)``:
    Must return an array-like *snapshot* of current values. In write-through
    wrappers, this is usually a materialization step (e.g. building a NumPy
    array from underlying objects).

Strongly recommended (common in AIMMD):
- ``__getitem__(self, key)``:
    Override if scalar indexing can be answered cheaply without building a full
    array.
- ``__setitem__(self, key, values)``:
    Implement if you want write-through behavior (e.g. assigning into the proxy
    updates underlying objects). This is required for the in-place operators
    implemented here (``+=``, ``-=`` etc.), because those operators use
    ``self[:] = ...``. :contentReference[oaicite:5]{index=5}

Indexing expectations:
- Your implementation should support integer indices and slicing; supporting
  NumPy-style fancy indexing is beneficial (AIMMD uses ``np.arange(...)[key]``
  patterns for this). :contentReference[oaicite:6]{index=6}

Broadcasting expectations (write-through proxies):
- When assigning ``obj[key] = values`` where ``key`` selects multiple elements,
  it is helpful to accept scalars and single-element arrays and broadcast them
  to the selected shape (as done in :class:`PathProperties`). :contentReference[oaicite:7]{index=7}

Performance notes
-----------------
- :meth:`_array` may be expensive if it materializes from many underlying
  objects. If so, prefer overriding :meth:`__getitem__` for scalar access.
- Many methods below call :meth:`_array` (``__len__``, ``__iter__``, comparisons,
  ``__repr__``, ``__getattr__``). If repeated materialization is expensive, the
  subclass may want to cache results externally (while ensuring freshness).

Notes
-----
This base class intentionally stays small and dependency-free (no NumPy import).
"""

# external
from abc import ABC, abstractmethod

# abstract array
class AbstractArray(ABC):
    """
    Abstract proxy that exposes an "array-like" interface.

    Subclasses must implement :meth:`_array` which returns the underlying
    array-like object used for *read* operations by default.

    In AIMMD this is often used for "write-through" property arrays: the object
    behaves like an array, but reads/writes are mapped to attributes stored in
    underlying domain objects (e.g., paths in a path ensemble). :contentReference[oaicite:8]{index=8}

    Minimal implementation
    ----------------------
    At minimum, implement :meth:`_array`. This enables:
    - ``len(obj)``,
    - iteration,
    - NumPy coercion via ``np.asarray(obj)``,
    - comparisons, and
    - attribute proxying (``obj.shape``, ``obj.dtype`` etc., if the backing array has them).

    Write-through / in-place operations
    -----------------------------------
    If you implement ``__setitem__``, then the in-place operators in this class
    (``__iadd__``, ``__isub__``, ``__imul__``, ``__itruediv__``) will work,
    because they compute a new array and assign it back via ``self[:] = new``.
    AIMMD's :class:`PathProperties` relies on exactly this pattern for setting
    properties in bulk (e.g., ``weights[:] = ...``). :contentReference[oaicite:9]{index=9}

    Overriding indexing
    -------------------
    You may override ``__getitem__`` to provide a more efficient scalar path
    (e.g., avoid constructing the full array for ``obj[i]``). If you do, keep
    slice/fancy-index behavior consistent with NumPy expectations (return an
    array-like result).
    """

    @abstractmethod
    def _array(self):
        """
        Return the backing array-like object.

        Returns
        -------
        array-like
            Any object that supports NumPy-style indexing and basic operations.

        Notes
        -----
        In many AIMMD proxies, this method **materializes** a NumPy array from
        underlying objects. If that is expensive, consider overriding
        :meth:`__getitem__` for scalar access. :contentReference[oaicite:10]{index=10}
        """
        # Must be implemented by subclasses.
        pass

    def __getitem__(self, key):
        # Default: delegate indexing to the backing array.
        # Subclasses may override for efficiency or write-through semantics.
        return self._array()[key]

    def __len__(self):
        # Delegate length to the backing array.
        return len(self._array())

    def __repr__(self):
        # Mirror the representation of the backing array for readability.
        return repr(self._array())

    def __array__(self, dtype=None):
        """
        NumPy coercion hook.

        This enables ``np.asarray(obj)`` to convert this object to a NumPy array.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            If provided, the returned array is cast to this dtype.

        Returns
        -------
        numpy.ndarray or array-like
            The backing array (possibly cast).
        """
        array = self._array()
        if dtype is not None:
            # Respect NumPy conventions: dtype forces a cast.
            array = array.astype(dtype)
        return array

    def __iter__(self):
        # Delegate iteration to the backing array.
        return self._array().__iter__()

    def __getattr__(self, attribute):
        # Proxy unknown attributes to the backing array.
        # This makes e.g. obj.shape, obj.dtype, obj.mean(), ... available.
        return getattr(self._array(), attribute)

    # ---- rich comparisons (forwarded) ----
    def __eq__(self, other):
        return self._array() == other

    def __ne__(self, other):
        return self._array() != other

    def __lt__(self, other):
        return self._array() < other

    def __le__(self, other):
        return self._array() <= other

    def __gt__(self, other):
        return self._array() > other

    def __ge__(self, other):
        return self._array() >= other

    # ---- reflected rich comparisons (forwarded) ----
    def __req__(self, other):
        return other == self._array()

    def __rne__(self, other):
        return other != self._array()

    def __rlt__(self, other):
        return other < self._array()

    def __rle__(self, other):
        return other <= self._array()

    def __rgt__(self, other):
        return other > self._array()

    def __rge__(self, other):
        return other >= self._array()

    def __bool__(self):
        # Match NumPy semantics: arrays are ambiguous in boolean context.
        # NOTE: The error message uses `{self.__class__.name}` literally because
        # it is not an f-string; this is kept unchanged by request.
        raise ValueError(
            "The truth value of an {self.__class__.name} "
            "is ambiguous. Use any() or all().")

    # ---- in-place arithmetic (requires __setitem__ in subclass) ----
    def __iadd__(self, other):
        # Compute result on the backing array, then write it back.
        # Requires that subclass supports slice assignment (self[:] = ...).
        new = self._array() + other
        self[:] = new
        return self

    def __isub__(self, other):
        new = self._array() - other
        self[:] = new
        return self

    def __imul__(self, other):
        new = self._array() * other
        self[:] = new
        return self

    def __itruediv__(self, other):
        new = self._array() / other
        self[:] = new
        return self

    def __invert__(self):
        # Bitwise invert (e.g., for boolean masks).
        return ~self._array()
