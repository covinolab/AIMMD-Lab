"""
aimmd.core.decorators
=====================

Small descriptor/decorator utilities used across AIMMD.

The main purpose of this module is to provide:

- :class:`class_or_instancemethod`:
    A method that can be called *either* on the class *or* on an instance,
    behaving like a ``classmethod`` when accessed via the class, and like a
    normal instance method when accessed via the instance.

- :class:`classproperty`:
    A read-only property evaluated on the class (similar to combining
    ``@property`` + ``@classmethod``), implemented as a descriptor.

These helpers are useful when AIMMD wants a unified API surface, for example:

- constructors / factories that can be invoked from class or instance,
- computed constants that depend on the class (e.g., configuration metadata).

Notes
-----
These are descriptors; the behavior is driven by Python's attribute access rules
and the ``__get__`` protocol.
"""

# aimmd imports
import functools

# class
class class_or_instancemethod(classmethod):
    """
    Descriptor behaving as both classmethod and instancemethod.

    When accessed on the class, it behaves like ``@classmethod``.
    When accessed on an instance, it behaves like a regular bound method.

    This is handy for APIs where the same call signature should work in both
    contexts without duplicating code.
    """

    def __get__(self, instance, owner):
        # If accessed on the class (instance is None), defer to classmethod.
        if instance is None:
            # Return the classmethod itself
            return super().__get__(instance, owner)
        else:
            # If accessed on an instance, bind the underlying function to it.
            # This emulates the standard function descriptor binding behavior.
            return self.__func__.__get__(instance, owner)


class classproperty:
    """
    Read-only property evaluated on the class.

    Usage
    -----
    >>> class A:
    ...     @classproperty
    ...     def x(cls):
    ...         return 123
    >>> A.x
    123

    Notes
    -----
    - This does not support setting/deleting (read-only).
    - The decorated function receives the *owner class* as its sole argument.
    """

    def __init__(self, func):
        # preserve original function metadata
        functools.update_wrapper(self, func)
        self.func = func

    def __get__(self, instance, owner):
        # Always call with the owner class; ignore instance.
        return self.func(owner)
