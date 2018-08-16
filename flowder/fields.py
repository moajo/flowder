from flowder.abstracts import Field
from flowder.processors import BuildVocab


def lowercase(tokenized):
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
            lower=False,
            vocab_processor=BuildVocab(),
            sos_token=None,
            eos_token=None,
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
        if eos_token is not None:
            loading_process.append(
                _add_eos(eos_token)
            )
        if sos_token is not None:
            loading_process.append(
                _add_sos(sos_token)
            )

        super(TextField, self).__init__(
            name,
            target_source,
            preprocess=preprocess,
            process=process,
            loading_process=loading_process,
        )
