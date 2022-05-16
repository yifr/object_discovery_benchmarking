import os
import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import nn
from data import movi
from tqdm import tqdm
from models import SIMONe
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
import faulthandler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')    

faulthandler.enable()

parser = ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, default="/mnt/fs6/yifr/simone/checkpoints")
parser.add_argument("--figure_dir", type=str, default="/mnt/fs6/yifr/simone/figures")
parser.add_argument("--log_dir", type=str, default="/mnt/fs6/yifr/simone/logs")
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--eval_every", type=int, default=100)
parser.add_argument("--checkpoint_every", type=int, default=5000)
parser.add_argument("--train_steps", type=int, default=1e5)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--warmup_steps", type=int, default=2500)
parser.add_argument("--plot_n_videos", type=int, default=1)
parser.add_argument("--movi", type=str, default="c")

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir, exist_ok=True)
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
if not os.path.exists(args.figure_dir):
    os.makedirs(args.figure_dir, exist_ok=True)

def learning_rate_update(optim, step, warmup_steps, max_lr, max_steps):
    """ Learning Rate Scheduler with linear warmup and cosine annealing

        Params:
        ------
        optim: torch.optim:
            Torch optimizer
        step: int:
            current training step
        warmup_steps: int:
            number of warmup steps
        max_lr: float:
            maximum learning rate
        max_steps: int:
            total number of training steps

        Returns:
        --------
        Updates optimizer and returns updated learning rate
    """
    if step < warmup_steps:
        warmup_percent_done = step / warmup_steps
        lr = max_lr * warmup_percent_done
        optim.lr = lr
    else:
        lr = 0.5 * (max_lr) * (1 + np.cos(step / max_steps * np.pi))
        optim.lr = lr

    return lr

def train(args, model, data, rank, step=0):

    writer = SummaryWriter(args.log_dir)
    def train_step(batch, model, optim):
        optim.zero_grad()
        images = batch["images"].to(rank)
        out = model(images)
        loss = out["loss/total"]
        loss.backward()
        optim.step()

        return out

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.log_dir)

    model.train()
    pbar = tqdm(total=args.train_steps)
    pbar.update(step)
    while step < args.train_steps:
        data.dataset.reset_iterator()
        for i, batch in enumerate(data):
            lr = learning_rate_update(
                optim, step, args.warmup_steps, args.lr, args.train_steps
            )
            out = train_step(batch, model, optim)
            step += 1
            pbar.update(step)
            if step % args.log_every == 0 and rank == 0:
                recons = out["recon"].detach().cpu().numpy()
                images = batch["images"].detach().cpu().numpy()
                loss = out["loss/total"].item()
                obj_kl_loss = out["loss/obj_kl"].item()
                frame_kl_loss = out["loss/frame_kl"].item()
                nll = out["loss/nll"].item()
                writer.add_scalar("train/total_loss", loss, step)
                writer.add_scalar("train/obj_kl_loss", obj_kl_loss, step)
                writer.add_scalar("train/frame_kl_loss", frame_kl_loss, step)
                writer.add_scalar("train/nll", nll, step)
                writer.add_video(
                    "train/input_video", images[:1], step
                )
                writer.add_video(
                    "train/reconstructed_video", recons[:1], step
                )

                print("Step: {}, Loss: {}".format(step, loss))
                print("          Obj KL: {}".format(obj_kl_loss))
                print("          Frame KL: {}".format(frame_kl_loss))
                print("          NLL: {}".format(nll))

            if step % args.eval_every == 0 and step > 0:
                # eval(model, data_loader, args, step, writer)
                
                if step % args.checkpoint_every == 0 and step > 0:
                    checkpoint = {"model": model.state_dict(),
                                  "optim": optim, "step": step}
                    if not os.path.exists(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir, exist_ok=True)
                    torch.save(
                        checkpoint,
                        os.path.join(args.checkpoint_dir,
                                     "checkpoint_{}.pth".format(step)),
                    )

            if step > args.train_steps:
                break

    print("Reached maximum training number of training steps...")

    checkpoint = {"model": model.state_dict(), "optim": optim, "step": step}
    torch.save(
        checkpoint,
        os.path.join(args.checkpoint_dir, "FINAL.pth".format(step)),
    )
    return


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)
    print(rank, world_size)
    torch.cuda.empty_cache()

    batch_size = 2
    n_frames = 10
    model = SIMONe.SIMONE((batch_size, n_frames, 3, 64, 64), device=rank).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    dataset_path = f"/mnt/fs6/honglinc/dataset/tensorflow_datasets/movi_{args.movi}/128x128/1.0.0"
    dataset = movi.MoviDataset(dataset_path, sequence_length=n_frames)
    num_workers = 0
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=num_workers, 
                            drop_last=False, shuffle=False, sampler=sampler)
    train(args, model, dataloader, rank)
    cleanup()

if __name__=="__main__":	
    world_size = 4
    mp.spawn(main,
             args=(world_size,),
             nprocs=4,
             join=True
             )
