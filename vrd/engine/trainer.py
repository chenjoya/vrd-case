import datetime
import logging
import os
import time
from tqdm import tqdm

import torch
import torch.distributed as dist

from vrd.data import make_data_loader
from vrd.utils.comm import get_world_size, synchronize, reduce_dict, is_main_process
from vrd.utils.metric_logger import MetricLogger
from vrd.engine.inference import inference
from vrd.utils.miscellaneous import mkdir

def recursive_to_device(batch, device):
    if isinstance(batch, (list,tuple)):
        batch = [recursive_to_device(b, device) for b in batch]
        return batch
    return batch.to(device)

def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    output_dir,
):
    logger = logging.getLogger("vrd.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH

    model.train()
    start_training_time = time.time()

    for epoch in range(arguments["epoch"], max_epoch + 1):
        max_iteration = len(data_loader)
        last_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch

        end = time.time()

        for _iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = _iteration + 1
            
            optimizer.zero_grad()
            batch = recursive_to_device(batch, device)
            loss_dict = model(batch)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            batch_time = time.time() - end
            
            if iteration % 100 == 0 or iteration == max_iteration:
                  meters.update(time=batch_time, data=data_time)
                  loss_dict_reduced = reduce_dict(loss_dict)
                  losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                  meters.update(loss=losses_reduced, **loss_dict_reduced)
                  eta_seconds = meters.time.global_avg * (max_iteration - iteration + last_epoch_iteration)
                  eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                  logger.info(
                      meters.delimiter.join(
                          [
                              "eta: {eta}",
                              "epoch: {epoch}/{max_epoch}",
                              "iteration: {iteration}/{max_iteration}",
                              "{meters}",
                              "lr: {lr:.6f}",
                              "max mem: {memory:.0f}",
                          ]
                      ).format(
                          eta=eta_string,
                          epoch=epoch,
                          max_epoch=max_epoch,
                          iteration=iteration,
                          max_iteration=max_iteration,
                          meters=str(meters),
                          lr=optimizer.param_groups[0]["lr"],
                          memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                      )
                  )
            end = time.time()
            
        scheduler.step()

        if epoch % checkpoint_period == 0:
            checkpointer.save(f"model_{epoch}e", **arguments)

        if data_loader_val is not None and test_period > 0 and epoch % test_period == 0:
                dataset_names = cfg.DATASETS.TEST
                dataset_name = dataset_names[0]
                output_folder = os.path.join(output_dir, "inference", dataset_name)
                save_json_basename = f"model_{epoch}e.json"
                save_json_file = os.path.join(output_folder, save_json_basename)
                mkdir(output_folder)
                synchronize()
                inference(
                    model,
                    data_loader_val[0],
                    dataset_name=dataset_name,
                    task=cfg.TASK,
                    save_json_file=save_json_file,
                    visualize_dir="", # do not visualize here
                    device=cfg.MODEL.DEVICE,
                )
                synchronize()
                model.train()
        
        
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
