import sys, os
from pathlib import Path

import numpy as np

from flowder.abstracts import Field
from flowder.sources import ImageSource

sys.path.append(os.pardir)
from flowder.fields import TextField
from flowder.utils import zip_source, file, create_dataset, directory, collect

iris = file("data/IRIS.csv").csv(header=None)

for data in iris:
    assert isinstance(data, tuple)
    first = iris[0]
    all(a == b for a, b in zip(first, data))
    break

for index, values in iris:
    assert np.issubdtype(type(index), np.integer)
    assert isinstance(values, dict)

d = directory("data/celebA/img_align_celeba")
# for p in d.item.suffix == ".jpg":
#     assert isinstance(p, Path)

anno = file("data/celebA/list_attr_celeba.txt").csv(header=1, sep="\s+")
assert len(anno) == 8
imgs = collect(anno.item[0], d.item.name, d).to(ImageSource)

# img = Field("img", process=mean(), postprocess=whitening())
# ds = imgs.create_datsset()
