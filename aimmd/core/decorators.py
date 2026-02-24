"""
...
"""

# aimmd imports
import functools

# class
class class_or_instancemethod(classmethod):
    def __get__(self, instance, owner):
        if instance is None:
            # Return the classmethod itself
            return super().__get__(instance, owner)
        else:
            # Bind the function to the instance
            return self.__func__.__get__(instance, owner)


class classproperty:
    def __init__(self, func):
        # preserve original function metadata
        functools.update_wrapper(self, func)
        self.func = func

    def __get__(self, instance, owner):
        return self.func(owner)
