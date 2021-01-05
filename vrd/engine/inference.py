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

def recursive_to_device(batch, device):
    if isinstance(batch, (list,tuple)):
        batch = [recursive_to_device(b, device) for b in batch]
        return batch
    return batch.to(device)

def compute_on_dataset(model, data_loader, device, timer):
    model.eval()
    outputs = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            timer.tic()
            batch = recursive_to_device(batch, device)
            output = model(batch)
            torch.cuda.synchronize()
            outputs.append(output)
            timer.toc()
    outputs = list(zip(*outputs))
    outputs = [o[0].extend_batch_eval(o) for o in outputs]
    return outputs

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    predictions = [p.all_gather(all_gather) for p in predictions_per_gpu]
    return predictions

def inference(
        model,
        data_loader,
        dataset_name,
        task,
        save_json_file,
        visualize_dir,
        device="cuda",
    ):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("vrd.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
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

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    if not is_main_process():
        return

    return evaluate(
        dataset=dataset,
        predictions=predictions,
        task=task,
        save_json_file=save_json_file,
        visualize_dir=visualize_dir
    )
