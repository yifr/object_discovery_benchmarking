import os
import glob
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from data import gestalt
from utils import metrics
from models import SIMONe
from models import SlotAttentionModels
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from argparse import ArgumentParser


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


def eval(model, data_loader, args, step, writer=None, save=True):
    """
    Runs eval loop on set set of data, logs results to tensorboard
    if writer is present
    Args:
        model: Model to evaluate
        data_loader: data loader
        args: args
        step: step to log
    """
    model.eval()
    data_loader.dataset.training = False

    print(f"Running Evaluation on {100} samples")
    with torch.no_grad():
        np.random.seed(args.seed)
        loss = 0
        fg_ari = 0
        mean_IOU = 0

        for i, batch in tqdm(enumerate(data_loader)):
            if i == 100:
                break
            images = batch["images"]
            flows = batch["flows"]
            if args.cue == "masks":
                # Only take first time step of a cue
                cue = batch[args.cue][:, :, 0]
            else:
                cue = batch[args.cue][:, 0]
            out = model(images, cues=cue)
            pred_flows = out["recon_combined"]
            loss += F.mse_loss(flows, pred_flows).item()

            pred_masks = out["masks"].detach()
            gt_masks = (
                batch["masks"].detach().sum(dim=3, keepdim=True)
            )  # Combine RGB channels into one
            B, N, T, C, H, W = gt_masks.shape
            gt_masks = gt_masks.reshape((B, N, T, H, W, C))
            pred_masks = pred_masks.transpose(1, 2)

            pred_groups = pred_masks.reshape(
                args.batch_size, N, -1).permute(0, 2, 1)
            true_groups = gt_masks.reshape(
                args.batch_size, N, -1).permute(0, 2, 1)
            fg_ari += metrics.adjusted_rand_index(
                true_groups, pred_groups).mean()

            gt_masks = gt_masks[:, 1:, ...].sum(
                dim=1
            )  # Combine individual mask slots and ignore backgrounds
            pred_masks = pred_masks.sum(dim=1)
            mean_IOU += metrics.mean_IOU(gt_masks, pred_masks)

            gt_masks = gt_masks.reshape(B, T, C, H, W)
            pred_masks = pred_masks.reshape(B, T, C, H, W)
            recons = out["recons"].sum(dim=2).reshape(B, T, 3, H, W)

        mean_IOU /= len(data_loader)
        fg_ari /= len(data_loader)
        loss /= len(data_loader)

        if writer is not None:
            writer.add_scalar("eval/loss", loss, step)
            writer.add_scalar("eval/fg_ari", fg_ari, step)
            writer.add_scalar("eval/mean_IOU", mean_IOU, step)

            writer.add_video("eval/input_video",
                             images[: args.plot_n_videos], step)
            writer.add_video("eval/pred_flow",
                             pred_flows[: args.plot_n_videos], step)
            writer.add_video("eval/gt_flow", flows[: args.plot_n_videos], step)
            writer.add_video("eval/pred_masks",
                             pred_masks[: args.plot_n_videos], step)
            writer.add_video(
                "eval/gt_masks", gt_masks[: args.plot_n_videos], step)
            writer.add_video("eval/slot_recons",
                             recons[: args.plot_n_videos], step)

        print("=" * 30 + " EVALUATION " + "=" * 30)
        print("Step: {}, Eval Loss: {}".format(step, i, loss))
        print("\tEval FG-ARI: {}, Eval Mean IOU: {}".format(fg_ari, mean_IOU))
        print("=" * 72)

    dataloader.dataset.training = True
    model.train()
    return


def train(model, data_loader, args, step=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def train_step(batch, model, optim):
        optim.zero_grad()
        images = batch["images"].to(device)

        out = model(images)
        losses = model.compute_loss(images, out)
        loss = losses["total_loss"]
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        loss.backward()
        optim.step()

        return out, losses

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.log_dir)

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.log_dir),
        record_shapes=True,
        with_stack=True
    )
    prof.start()

    while step < args.train_iters:

        for i, batch in enumerate(tqdm(data_loader)):
            lr = learning_rate_update(
                optim, step, args.warmup_steps, args.lr, args.train_iters)
            out, losses = train_step(batch, model, optim)
            if i % args.log_every == 0:
                recons = out["recons"].detach().cpu().numpy()
                images = batch["images"].detach().cpu().numpy()

                loss = losses["total_loss"].item()
                obj_kl_loss = losses["obj_kl_loss"].item()
                frame_kl_loss = losses["frame_kl_loss"].item()
                writer.add_scalar("train/total_loss", loss, step)
                writer.add_scalar("train/obj_kl_loss", obj_kl_loss, step)
                writer.add_scalar("train/frame_kl_loss", frame_kl_loss, step)

                writer.add_video(
                    "train/input_video", images[: args.plot_n_videos], step
                )
                writer.add_video(
                    "train/reconstructed_video", recons[: args.plot_n_videos], step
                )

                print("Step: {}, Loss: {}".format(step, loss))
                prof.step()

            if step % args.eval_every == 0 and step > 0:
                # eval(model, data_loader, args, step, writer)

                checkpoint = {"model": model.state_dict(),
                              "optim": optim, "step": step}
                if not os.path.exists(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save(
                    checkpoint,
                    os.path.join(args.checkpoint_dir,
                                 "checkpoint_{}.pth".format(step)),
                )

            step += 1

    prof.stop()
    print("Reached maximum training number of training steps...")
    eval(model, data_loader, args, step, writer)

    checkpoint = {"model": model.state_dict(), "optim": optim, "step": step}
    torch.save(
        checkpoint,
        os.path.join(args.checkpoint_dir, "FINAL.pth".format(step)),
    )
    return


def load_latest(args):
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_*.pth")
    checkpoints = glob.glob(checkpoint_path)
    if len(checkpoints) == 0:
        return None
    checkpoints.sort()
    return torch.load(checkpoints[-1])


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model params
    parser.add_argument("--model", type=str,
                        default="SIMONE", help="SIMONE or SAVI")

    # Data params
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames to train on")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size")

    # Training params
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--warmup_steps", type=int, default=2500,
                        help="Warmup steps for learning rate")
    parser.add_argument("--grad_clip", type=float,
                        default=0.05, help="Gradient Clipping")
    parser.add_argument(
        "--train_iters", type=int, default=50e4, help="Number of training steps"
    )
    parser.add_argument("--log_every", type=int, default=10,
                        help="How often to log losses")
    parser.add_argument(
        "--eval_every", type=int, default=1000, help="How often to run eval"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use multiple GPUs")

    # Paths
    parser.add_argument(
        "--plot_n_videos", type=int, default=4, help="Number of videos to plot"
    )
    parser.add_argument(
        "--log_dir", type=str, default="/om2/user/yyf/GestaltVision/runs/SIMONe/batch_size=1_frames=10"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="/om2/user/yyf/GestaltVision/models/SIMONe/batch_size=1_frames=10"
    )
    parser.add_argument("--data_dir", type=str,
                        default="/om/user/yyf/CommonFate/scenes")
    parser.add_argument(
        "--top_level",
        type=str,
        nargs="+",
        default=["voronoi", "noise"],
        help="texture split",
    )
    parser.add_argument(
        "--sub_level",
        type=str,
        nargs="+",
        default=["superquadric_1", "superquadric_2", "superquadric_3"],
        help="object split",
    )

    parser.add_argument(
        "--load_latest_model",
        action="store_true",
        help="Continue training from latest checkpoint",
    )
    args = parser.parse_args()

    dataloader = DataLoader(
        gestalt.Gestalt(
            root_dir=args.data_dir,
            top_level=args.top_level,
            sub_level=args.sub_level,
            frames_per_scene=args.num_frames,
            train_split=0.95,
            passes=["images"]
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    ex = next(iter(dataloader))["images"]
    print(ex.shape)

    if args.model == "SIMONE":
        model = SIMONe.SIMONE(ex.shape, 128)
    elif args.model == "SAVI":
        model = SlotAttentionModels.SAVi()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.multi_gpu:
        dataloader = nn.DataParallel(dataloader)
        model = nn.DataParallel(model)
    model.to(device)

    step = 0
    if args.load_latest_model:
        checkpoint = load_latest(args)
        if checkpoint:
            model_weights = checkpoint.get("model")
            model.load_state_dict(model_weights)
            step = checkpoint["step"]
            print("LOADED MODEL WEIGHTS. STARTING TRAINING AT STEP: ", step)
        else:
            print("NO MODEL FOUND ==> STARTING TRAINING FROM SCRATCH")
            sys.exit()

    train(model, dataloader, args, step)
