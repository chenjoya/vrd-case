import torch

class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, is_train, task):
        self.is_train = is_train
        self.task = task
    
    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images, batch_instances, batch_relationships = transposed_batch
        return images, batch_instances, batch_relationships
