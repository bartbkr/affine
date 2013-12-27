"""
These are utilies used by the affine model class
"""

import sys

from numpy.linalg import LinAlgError

def retry(func, attempts):
    """
    Decorator that attempts a function multiple times, even with exception
    """
    def inner_wrapper(*args, **kwargs):
        for attempt in xrange(attempts):
            try:
                return func(*args, **kwargs)
                break
            except LinAlgError:
                print "Trying again, maybe bad initial run"
                print "LinAlgError:", sys.exc_info()[0]
                continue
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise
    return inner_wrapper

