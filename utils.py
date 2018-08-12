from abstracts import Field
from sets import TextFileSource, ZipSource, Dataset


def file(path):
    return TextFileSource(path)


def zip_source(*sources):
    return ZipSource(*sources)


def create_dataset(size, *fields, return_as_tuple=False):
    assert isinstance(size, int)
    assert all(isinstance(f, Field) for f in fields)
    return Dataset(
        fields,
        size,
        return_as_tuple=return_as_tuple
    )
