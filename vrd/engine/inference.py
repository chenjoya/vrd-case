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
    outputs, indicators, idxs = [], [], []
    for images, targets, _idxs in tqdm(data_loader):
        with torch.no_grad():
            timer.tic()
            _outputs = model(images.to(device)) # no targets
            torch.cuda.synchronize()
            timer.toc()
            outputs.append(_outputs)
            idxs.append(_idxs.to(device))
            indicators.append(targets['indicators'].to(device))
    idxs = torch.cat(idxs)
    indicators = torch.cat(indicators)
    if isinstance(outputs[0], (list, tuple)): # has multiple returns
        outputs = list(zip(*outputs))
        outputs = [torch.cat(o) for o in outputs]
    else: # is a tensor
        outputs = [torch.cat(outputs)]
    return outputs, indicators, idxs

def split_by_indicators(predictions, indicators):
    begin = 0
    segments = []
    for i in indicators:
        end = begin + i.item()
        segments.append(predictions[begin:end])
        begin = end
    return segments

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, indicators_per_gpu, idxs_per_gpu, length):
    all_predictions = [all_gather(p) for p in predictions_per_gpu]
    all_indicators = all_gather(indicators_per_gpu)
    all_idxs = all_gather(idxs_per_gpu)
    if not is_main_process():
        return
    if all_idxs.numel() != all_idxs.max() + 1:
        logger = logging.getLogger("vrd.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
        # may not size div
        all_indicators = all_indicators[:length]
        all_idxs = all_idxs[:length]
        
    all_predictions = [split_by_indicators(p, all_indicators) for p in all_predictions]
    _, idxs = all_idxs.sort()
    all_predictions = [[p[i].cpu() for i in idxs] for p in all_predictions]
    return all_predictions

def inference(
        model,
        data_loader,
        dataset_name,
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
    predictions, indicators, idxs = compute_on_dataset(model, data_loader, device, inference_timer)
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

    predictions = _accumulate_predictions_from_multiple_gpus(predictions, indicators, idxs, len(dataset))
    if not is_main_process():
        return

    return evaluate(dataset=dataset,
        predictions=predictions,
        save_json_file=save_json_file,
        visualize_dir=visualize_dir)
