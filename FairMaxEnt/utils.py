"""
Common utility functions
"""

import hashlib

def hash(*args):
    """
        Creates a unique name for given set of arguments
        takes arbitrary list of objects
        returns a string
    """
    hashSHA512 = hashlib.sha512()
    hashSHA512.update(str(args).encode("utf-8"))
    return hashSHA512.hexdigest()