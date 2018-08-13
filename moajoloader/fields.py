from moajoloader.abstracts import Field
from moajoloader.processors import BuildVocab


def lowercase(tokenized):
    return [word.lower() for word in tokenized]


def _include_length(data):
    return data, len(data)


def _numericalize(vocab_processor):
    def wrapper(data):
        return [vocab_processor.vocab.stoi[word] for word in data]

    return wrapper


# def numericalize_old(self, arr, device=None, train=True):
#     """Turn a batch of examples that use this field into a Variable.
#
#     If the field has include_lengths=True, a tensor of lengths will be
#     included in the return value.
#
#     Arguments:
#         arr (List[List[str]], or tuple of (List[List[str]], List[int])):
#             List of tokenized and padded examples, or tuple of List of
#             tokenized and padded examples and List of lengths of each
#             example if self.include_lengths is True.
#         device (-1 or None): Device to create the Variable's Tensor on.
#             Use -1 for CPU and None for the currently active GPU device.
#             Default: None.
#         train (boolean): Whether the batch is for a training set.
#             If False, the Variable will be created with volatile=True.
#             Default: True.
#     """
#     if self.include_lengths and not isinstance(arr, tuple):
#         raise ValueError("Field has include_lengths set to True, but "
#                          "input data is not a tuple of "
#                          "(data batch, batch lengths).")
#     if isinstance(arr, tuple):
#         arr, lengths = arr
#         lengths = torch.LongTensor(lengths)
#
#     if self.use_vocab:
#         if self.sequential:
#             arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
#         else:
#             arr = [self.vocab.stoi[x] for x in arr]
#
#         if self.postprocessing is not None:
#             arr = self.postprocessing(arr, self.vocab, train)
#     else:
#         if self.tensor_type not in self.tensor_types:
#             raise ValueError(
#                 "Specified Field tensor_type {} can not be used with "
#                 "use_vocab=False because we do not know how to numericalize it. "
#                 "Please raise an issue at "
#                 "https://github.com/pytorch/text/issues".format(self.tensor_type))
#         numericalization_func = self.tensor_types[self.tensor_type]
#         # It doesn't make sense to explictly coerce to a numeric type if
#         # the data is sequential, since it's unclear how to coerce padding tokens
#         # to a numeric type.
#         if not self.sequential:
#             arr = [numericalization_func(x) if isinstance(x, six.string_types)
#                    else x for x in arr]
#         if self.postprocessing is not None:
#             arr = self.postprocessing(arr, None, train)
#
#     arr = self.tensor_type(arr)
#     if self.sequential and not self.batch_first:
#         arr.t_()
#     if device == -1:
#         if self.sequential:
#             arr = arr.contiguous()
#     else:
#         arr = arr.cuda(device)
#         if self.include_lengths:
#             lengths = lengths.cuda(device)
#     if self.include_lengths:
#         return Variable(arr, volatile=not train), lengths
#     return Variable(arr, volatile=not train)


class TextField(Field):
    """
    文字列データに対する処理
    """

    def __init__(
            self,
            name,
            target_source,
            tokenize=lambda s: s.split(),
            lower=False,
            vocab_processor=BuildVocab(),
            include_length=False,
            numericalize=True,
    ):
        preprocess = []
        if tokenize:
            preprocess.append(tokenize)
        if lower:
            preprocess.append(lowercase)

        process = []
        if vocab_processor:
            process.append(vocab_processor)

        loading_process = []
        if numericalize:
            assert vocab_processor
            loading_process.append(_numericalize(vocab_processor))
        if include_length:
            loading_process.append(_include_length)

        super(TextField, self).__init__(
            name,
            target_source,
            preprocess=preprocess,
            process=process,
            loading_process=loading_process,
        )
