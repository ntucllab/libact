
def inherit_docstring_from(cls):
    """
    Decorator for class methods to inherit docstring from :code:`cls`.
    """
    # https://groups.google.com/forum/#!msg/comp.lang.python/HkB1uhDcvdk/lWzWtPy09yYJ
    def docstring_inheriting_decorator(fn):
        fn.__doc__ = getattr(cls,fn.__name__).__doc__
        return fn
    return docstring_inheriting_decorator

__all__ = ['inherit_docstring_from']
