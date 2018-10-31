import hashlib
import sys
from collections import Hashable


def default_hash_func(obj, hash_func=None):
    if hash_func is None:
        hash_func = default_hash_func
    if type(obj) == str:
        # note: default string hash will change to different value at the next session
        return int(hashlib.sha1(obj.encode('utf-8')).hexdigest(), 16) % sys.maxsize
    if isinstance(obj, Hashable):
        return hash(obj) % sys.maxsize
    if isinstance(obj, list):
        hs = 1
        for a in obj:
            hs = (hs * 31 + hash_func(a)) % sys.maxsize
        return hs
    if isinstance(obj, dict):
        hs = 1
        for k, v in obj.items():
            hs = (hs * 31 + hash_func(k)) % sys.maxsize
            hs = (hs * 31 + hash_func(v)) % sys.maxsize
        return hs
    raise ValueError(
        f"{obj} is not hashable.\nall arguments must be hashable"
    )


def extended_hash_func(extension: dict):
    def wrapper(obj):
        if type(obj) in extension:
            hashable = extension[type(obj)](obj)
            return default_hash_func(hashable, hash_func=wrapper)
        if "*" in extension:
            hashable = extension["*"](obj)
            return default_hash_func(hashable, hash_func=wrapper)

        return default_hash_func(obj, hash_func=wrapper)

    return wrapper


to_string_hash_func = extended_hash_func({
    "*": lambda a: str(a)
})
