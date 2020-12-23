import torch

def stack_dict(_ds):
    ds = {}
    for k in _ds[0].keys():
        ds[k] = torch.stack([_d[k] for _d in _ds])
    return ds

def cat_dict(_ds):
    ds = {}
    for k in _ds[0].keys():
        ds[k] = torch.cat([_d[k] for _d in _ds])
    return ds

class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, is_train, tasks):
        self.is_train = is_train
        self.tasks = tasks

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images, targets, idxs = transposed_batch
        if ("OBJCLS" in self.tasks) or ("RELCLS" in self.tasks): # multiple images
            # indicator to distinguish idx
            indicators = torch.tensor([len(i) for i in images])
            images = torch.cat(images)
            targets = cat_dict(targets)
            targets['indicators'] = indicators
        else:
            images = torch.stack(images)
            targets = stack_dict(targets)
        return images, targets, torch.stack(idxs) 
