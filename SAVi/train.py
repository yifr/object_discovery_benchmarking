import os
import argparse
from tdw_dataset import *
from raft_eval import *
from SlotAttentionModels import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from miou_metric import measure_miou_metric
from flow_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()


parser.add_argument('--run_name', default='tdw1_recon=flow_k=7', type=str)
parser.add_argument('--recon_regime', default="flow", type=str, help="what's being reconstructed")
parser.add_argument("--output_channels", default=3, type=int, help="how many channels to output")
parser.add_argument('--model_dir', default='/om2/user/yyf/ECCV_2022/models/slot-attention/', type=str, help='where to save models' )
parser.add_argument('--figure_dir', default='/om2/user/yyf/ECCV_2022/figures/slot-attention/')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0002, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of epochs')
parser.add_argument('--log_every', default=500, type=int, help='how often to log losses')
parser.add_argument('--ckpt_freq', default=10000, type=int, help='how often to save model')
parser.add_argument('--load_latest_ckpt', action="store_true")
opt = parser.parse_args()
resolution = (512, 512)
torch.manual_seed(opt.seed)

def load_latest():
    checkpoint_dir = os.path.join(opt.model_dir, opt.run_name)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_step=*.pt")
    checkpoints = glob.glob(checkpoint_path)
    if len(checkpoints) == 0:
        return None
    sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split(".")[0].split("=")[-1]))
    print("Loading checkpoint: ", sorted_checkpoints[-1])
    return torch.load(sorted_checkpoints[-1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up data
tdw_dataset_dir = "/om2/user/yyf/tdw_playroom_small"
raft_eval_path = "../RAFT/models/raft-sintel.pth"
flow_threshold = 0.5
train_dataloader = DataLoader(TDWDataset(tdw_dataset_dir, training=True),
                         batch_size=opt.batch_size,
                         shuffle=True)

model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim,
                                 output_channels=opt.output_channels).to(device)
if opt.load_latest_ckpt:
    ckpt = load_latest()
    weights = ckpt["state_dict"]
    model.load_state_dict(weights)

criterion = nn.MSELoss()

params = [{'params': model.parameters()}]


optimizer = optim.Adam(params, lr=opt.learning_rate)

start = time.time()
i = 0
steps_logged = []
losses = []
for epoch in range(opt.num_epochs):
    model.train()

    total_loss = 0

    for batch in tqdm(train_dataloader):
        i += 1
        image_1, segment_map, gt_moving, raft_moving = batch
        image_1 = image_1.cuda()
        if opt.recon_regime == "rgb":
            image = image_1 / 255.
            target = image
        elif opt.recon_regime == "rgb_flow":
            image = image_1 / 255.
            flow = raft_moving.unsqueeze(1).cuda()
            target = torch.cat([image, flow], 1)
        elif opt.recon_regime == "flow":
            image = image_1 / 255.
            target = raft_moving.unsqueeze(1).cuda().float()
        else:
            raise ValueError(f"{opt.recon_regime} not recognized!")

        """
        if i < opt.warmup_steps:
            learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        else:
            learning_rate = opt.learning_rate

        learning_rate = learning_rate * (opt.decay_rate ** (
           i / opt.decay_steps))
        optimizer.param_groups[0]['lr'] = opt.learning_rate
        """
        recon_combined, recons, masks, slots = model(image)
        loss = criterion(recon_combined, target)
        total_loss += loss.item()
        if (i + 1) % opt.log_every == 0:

            losses.append(loss.item())
            steps_logged.append(i)
            run_dir = os.path.join(opt.figure_dir, opt.run_name)
            if not os.path.exists(run_dir):
                os.makedirs(run_dir, exist_ok=True)

            figure_prefix = os.path.join(run_dir, f"{opt.run_name}_step={i:05d}_")

            fig, axs = plt.subplots(1, 2, figsize=(12, 8))

            plot_idx = 0
            pred = recon_combined[plot_idx].detach().cpu().numpy().transpose(1, 2, 0)
            target = target[plot_idx].detach().cpu().numpy().transpose(1, 2, 0)
            if opt.recon_regime == "rgb_flow":
                pred = pred[:, :, :-1] # reconstruct RGB
                target = target[:, :, :-1]
                # pred = flow_to_image(pred)
                # target = flow_to_image(target)

            axs[0].imshow(target)
            axs[1].imshow(pred)
            axs[0].set_title("Target")
            axs[1].set_title("Predicted")
            plt.tight_layout()
            plt.savefig(figure_prefix + "reconstruction.png")

            fig, ax = plt.subplots()
            ax.plot(steps_logged, losses)
            ax.set_title("Loss Curve")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Loss")
            plt.savefig(os.path.join(run_dir, f"{opt.run_name}_loss.png"))
            plt.close('all')

        del recons, masks, slots

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % opt.ckpt_freq == 0:
            save_dir = os.path.join(opt.model_dir, opt.run_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"checkpoint_step={i+1}.pt")
            torch.save({
                'state_dict': model.state_dict(),
                }, save_path)

    total_loss /= len(train_dataloader)

    print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
        datetime.timedelta(seconds=time.time() - start)))

