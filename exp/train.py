import time
import logging

import torch

from utils import get_remaining_time


logger = logging.getLogger(__name__)


def train(model, data_loader, optimizer, scheduler, device, epoch, args):
    torch.cuda.reset_peak_memory_stats(device)

    model.train()

    end = time.time()
    for iter, (image, label, _) in enumerate(data_loader):
        start = time.time()
        toprint = f"Epoch: [{epoch}|{args.schedule_config['train_epochs']}], "
        toprint += f"Iter: [{args.schedule_config['curr_iter']}|{args.schedule_config['train_iters']}], "
        toprint += f"Data Time: {(start - end):.6f}, "

        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        loss = model.forward_loss(image, label)
        loss.backward()
        optimizer.step()
        scheduler.step()

        end = time.time()
        if iter == 0 or (args.schedule_config["curr_iter"] % args.schedule_config["log_iters"] == 0):
            remaining_time = get_remaining_time(iter, epoch, len(data_loader), end, args)
            mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            lr = optimizer.param_groups[0]["lr"]

            toprint += f"Batch Time: {(end - start):.6f}, "
            toprint += f"Remaining Time: {remaining_time}, "
            toprint += f"Memory: {mem:.2f}, "
            toprint += f"Learning Rate: {lr:.6f}, "
            toprint += f"Loss({loss.item():.6f})"
            logger.info(toprint)

        args.schedule_config["curr_iter"] += 1
        if args.schedule_config["curr_iter"] > args.schedule_config["train_iters"]:
            break
