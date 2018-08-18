from .abstracts import Field
from .processors import BuildVocab


def _lowercase(tokenized):
    return [word.lower() for word in tokenized]


def _add_sos(sos_token):
    def wrapper(data: list):
        return [sos_token] + data

    return wrapper


def _add_eos(eos_token):
    def wrapper(data: list):
        return data + [eos_token]

    return wrapper


class TextField(Field):
    """
    文字列データに対する処理
    """

    def __init__(
            self,
            name,
            target_source,
            tokenize=lambda s: s.split(),
            lowercase=False,
            vocab_processor=BuildVocab(),
            sos_token=None,
            eos_token=None,
            numericalize=True,
    ):
        preprocess = []
        if tokenize:
            preprocess.append(tokenize)
        if lowercase:
            preprocess.append(_lowercase)

        process = []
        if vocab_processor:
            process.append(vocab_processor)

        loading_process = []
        if eos_token is not None:
            loading_process.append(
                _add_eos(eos_token)
            )
        if sos_token is not None:
            loading_process.append(
                _add_sos(sos_token)
            )
        if numericalize:
            assert vocab_processor is not None
            loading_process.append(
                vocab_processor.numericalize
            )

        super(TextField, self).__init__(
            name,
            target_source,
            preprocess=preprocess,
            process=process,
            loading_process=loading_process,
        )
