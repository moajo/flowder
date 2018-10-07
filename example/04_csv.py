from pathlib import Path
import numpy as np
from PIL import Image

from flowder.pipes import select
from flowder.source.base import filtered, mapped
from flowder.utils import directory
from flowder.utils.csv import csv
from flowder.utils.image import to_image

iris = csv("data/IRIS.csv", header=None)

for data in iris:
    assert isinstance(data, tuple)
    first = iris[0]
    all(a == b for a, b in zip(first, data))
    break

for index, values in iris:
    assert np.issubdtype(type(index), np.integer)
    assert isinstance(values, dict)

images_dir_path = Path("data/celebA/img_align_celeba")
d = directory(images_dir_path)
for p in d | filtered(lambda a: a.suffix == ".jpg"):
    assert isinstance(p, Path)

for p in d | filtered(lambda a: a.suffix == ".jpg"):
    assert isinstance(p, Path)

anno = csv("data/celebA/list_attr_celeba.txt", header=1, sep="\s+")
assert len(anno) == 8
imgs = anno | select(0) | mapped(lambda name: images_dir_path / name) | to_image()

# img = Field("img", process=mean(), postprocess=whitening())

for img in imgs:
    assert isinstance(img, Image.Image)
