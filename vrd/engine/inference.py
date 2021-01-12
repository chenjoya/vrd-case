import logging
import time
import os
from tqdm import tqdm

import torch

from vrd.data.datasets.evaluate import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, timer):
    model.eval()
    outputs = []
    for batches in tqdm(data_loader):
        with torch.no_grad():
            timer.tic()
            batches = [[b.to(device) for b in batch] for batch in batches]
            output = model(batches)
            torch.cuda.synchronize()
            outputs.append(output)
            timer.toc()
    outputs = list(zip(*outputs))
    # flatten
    outputs = [sum(output, []) for output in outputs]
    return outputs

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    predictions = [[] for _ in predictions_per_gpu]
    for item, prediction_per_gpu in enumerate(predictions_per_gpu):
        for p in prediction_per_gpu:
            predictions[item] += p.all_gather(all_gather)
    return predictions

def inference(
        model,
        data_loader,
        task,
        visualize_dir,
        device="cuda",
    ):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("vrd.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset.__class__.__name__, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / inference per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    evaluate(
        dataset=dataset,
        predictions=predictions,
        task=task,
        visualize_dir=visualize_dir
    )

    # predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    #if not is_main_process():
    #    return
    
    #return 
