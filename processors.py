class Processor:
    """
    前の要素から値を受け取って値を帰す関数。Fieldはこれを使って処理する
    """

    def __init__(self, left, right):
        assert left is Processor or left is None
        assert right is Processor or right is None
        self.left = left
        self.right = right

    def get_root(self):
        if self.left is None:
            return self
        return self.left.get_root()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def pipe(self, parent_processor):
        pass  # TODO

        # def get_value(self, i):
        raise NotImplementedError()

    def calculate_value(self, parent_value):
        raise NotImplementedError()


class RawProcessor(Processor):

    def __init__(self, target_set):
        super(RawProcessor, self).__init__(left=None, right=None)
        self.target_set = target_set

    def __call__(self, data):
        return data

    # def get_value(self, i):
    #     return self.target_set[i]
