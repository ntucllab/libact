
import sys

import numpy as np

# Syntax sugar.
_VER = sys.version_info
#: Python 2.x?
IS_PY2 = (_VER[0] == 2)
#: Python 3.x?
IS_PY3 = (_VER[0] == 3)

if IS_PY2:
    from future_builtins import zip

__all__ = ['inherit_docstring_from', 'seed_random_state', 'zip']

zip = zip


def inherit_docstring_from(cls):
    """Decorator for class methods to inherit docstring from :code:`cls`
    """
    # https://groups.google.com/forum/#!msg/comp.lang.python/HkB1uhDcvdk/lWzWtPy09yYJ
    def docstring_inheriting_decorator(fn):
        fn.__doc__ = getattr(cls, fn.__name__).__doc__
        return fn
    return docstring_inheriting_decorator


def seed_random_state(seed):
    """Turn seed into np.random.RandomState instance
    """
    if (seed is None) or (isinstance(seed, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r can not be used to generate numpy.random.RandomState"
                     " instance" % seed)
