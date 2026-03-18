Path Package
============

Primary Class
-------------

:class:`aimmd.Path` represents a trajectory-backed path assembled from one or
more file segments.

Internal Modules
----------------

:mod:`aimmd.path._helpers`
   Construction and indexing setup.

:mod:`aimmd.path._properties`
   Derived properties such as frame counts, filenames, and type information.

:mod:`aimmd.path._methods`
   Higher-level operations on paths.

:mod:`aimmd.path._extract` and :mod:`aimmd.path._get`
   Data extraction from readers, cached arrays, and multi-segment paths.

:mod:`aimmd.path._positions`
   Convenience selection helpers for initial, final, middle, and shooting
   frames.

:mod:`aimmd.path._compute`
   Batch computation and caching of states, descriptors, values, and other
   per-frame arrays.

:mod:`aimmd.path._io`
   Path extension and writing to trajectory files.

:mod:`aimmd.path._magic`
   Indexing and operator behavior.

Supporting Modules
------------------

:mod:`aimmd.path.chainreader`
   Reader-like adapter for traversing multiple readers as one logical sequence.

:mod:`aimmd.path.utils`
   Helper functions for filenames, cache naming, and batched path-level work.

See Also
--------

The full autodoc pages for these modules live in :doc:`lowlevel/path`.
