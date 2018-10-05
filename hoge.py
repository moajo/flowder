#!/usr/bin/env python
from rx import Observable
from tqdm import tqdm

from flowder.source import Source

a = list(tqdm(Source(Observable.from_(range(10)))))
print(a)