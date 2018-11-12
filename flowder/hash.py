import hashlib
from collections import Hashable

hash_max_size = 2 ** 31 - 1


def default_hash_func(obj, extension: dict = None):
    if extension is not None:
        for ty in extension.keys():
            if isinstance(obj, ty):
                return default_hash_func(
                    extension[ty](obj),
                    extension=extension)
    if type(obj) == str:
        # note: default string hash will change to different value at the next session
        return int(hashlib.sha1(obj.encode('utf-8')).hexdigest(), 16) % hash_max_size
    if isinstance(obj, Hashable):
        return hash(obj) % hash_max_size
    if isinstance(obj, list):
        hs = 1
        for a in obj:
            hs = (hs * 31 + default_hash_func(a, extension=extension)) % hash_max_size
        return hs
    if isinstance(obj, dict):
        hs = 1
        for k, v in obj.items():
            hs = (hs * 31 + default_hash_func(k, extension=extension)) % hash_max_size
            hs = (hs * 31 + default_hash_func(v, extension=extension)) % hash_max_size
        return hs
    if extension is not None and "*" in extension:
        return default_hash_func(extension["*"](obj), extension=extension)
    raise ValueError(
        f"{obj} is not hashable.\nall arguments must be hashable"
    )


def extended_hash_func(extension):
    def wrapper(obj):
        return default_hash_func(obj, extension)

    return wrapper


to_string_hash_func = extended_hash_func({
    "*": lambda a: str(a)
})
