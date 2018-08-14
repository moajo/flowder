from flowder import Source


class HookSource(Source):
    def __init__(self, parent, getitem_callback, iter_callback):
        super(HookSource, self).__init__(parent)
        self.getitem_callback = getitem_callback
        self.iter_callback = iter_callback

    def calculate_size(self):
        return len(self.parents[0])

    def __getitem__(self, item):
        self.getitem_callback(item)
        return self.parents[0][item]

    def __iter__(self):
        self.iter_callback()
        for v in self.parents[0]:
            yield v




