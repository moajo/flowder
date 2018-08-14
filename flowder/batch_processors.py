import torch


def sort(sort_key):
    def wrapper(batch):
        try:
            return sorted(batch, key=sort_key)
        except KeyError:
            raise KeyError("Failed to sort batch: is sort_key correct?")

    return wrapper


def to_device(device):
    def wrapper(batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            return tuple(wrapper(b, device) for b in batch)
        if isinstance(batch, dict):
            return {key: wrapper(batch[key], device) for key in batch}
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch

    return wrapper
