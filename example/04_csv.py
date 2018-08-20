from pathlib import Path
import numpy as np
from flowder.source import ImageSource
from flowder import file, directory

iris = file("data/IRIS.csv").csv(header=None)

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
for p in d.item_if.suffix == ".jpg":
    assert isinstance(p, Path)

for p in d.item_if.suffix == ".jpg":
    assert isinstance(p, Path)

anno = file("data/celebA/list_attr_celeba.txt").csv(header=1, sep="\s+")
assert len(anno) == 8
imgs = anno.item[0].map(lambda name: images_dir_path / name).to(ImageSource)

# img = Field("img", process=mean(), postprocess=whitening())
ds = imgs.create()
