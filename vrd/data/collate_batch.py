import torch

def pack(ds):
    return [d for d in ds if d is not None]

def get_intervals(ds):
    begin = 0
    intervals = []
    for i in ds:
        end = begin + len(i)
        intervals.append([begin, end])
        begin = end
    return intervals

class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, is_train, task):
        self.is_train = is_train
        self.task = task
    
    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        #return getattr(self, "call_" + self.task.lower())(transposed_batch)
        images, instances, relationships = transposed_batch
        instances = instances[0].extend(instances)
        relationships = relationships[0].extend(relationships)
        return images, instances, relationships
