import sys, os

sys.path.append(os.pardir)
from fields import TextField
from sources import Example
from utils import zip_source, file, create_dataset

iris = file("data/IRIS.csv").csv()

for data in iris:
    assert isinstance(data, list)
    first = iris[0]
    all(a == b for a, b in zip(first, data))
    break

for column1 in iris.item[0]:
    assert isinstance(column1, str)
