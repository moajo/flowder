#!/usr/bin/env python
from pathlib import Path
import zipfile
import io

from flowder import from_array


def zip_file(zip_file_path: Path, content_file: str):
    assert isinstance(zip_file_path, Path)
    assert isinstance(content_file, str)
    z = zipfile.ZipFile(zip_file_path)
    ss = io.BytesIO(z.read(content_file))
    s = io.TextIOWrapper(ss)
    ls = [a[:-1] for a in s.readlines()]
    return from_array(ls)
