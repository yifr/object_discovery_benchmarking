import os
import torch
import einops as E
from models import networks
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class Encoder(nn.Module):
    """
    Outputs frame encoding for each frame in a batch (B, T, H, W, C)
    Frame latents are given 3D position encodings to maintain
    temporal order before getting sent through transformer blocks.
    """

    def __init__(self, input_size, num_slots=16, z_dim=32, feature_dim=128):
        super(Encoder, self).__init__()
        b, t, c, h, w = input_size
        self.num_slots = num_slots
        self.z_dim = z_dim

        # Number of layers in conv net is dependent on resolution
        num_layers = np.log2(w / 8)
        self.ConvBlock = networks.ConvNet(num_layers,
                                          in_channels=c,
                                          mid_channels=feature_dim,
                                          out_channels=feature_dim)

        self.position_embedding = networks.PositionalEncodingPermute3D(128)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.0,
        )
        self.transformerA = nn.TransformerEncoder(transformer_encoder_layer, 4)
        self.transformerB = nn.TransformerEncoder(transformer_encoder_layer, 4)

        # MLP from objects --> frames
        self.lambda_frame_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )

        # MLP from frames --> objects
        self.lambda_obj_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        assert h == w, f"Height and Width should be equal for input shape: {x.shape}"

        # flatten input along time dim
        x = E.rearrange(x, "b t c h w -> (b t) c h w")

        # output of conv encoder should be b, t, 128, 8, 8
        z = self.ConvBlock(x)
        z = E.rearrange(z, "(b t) c h w -> b t c h w", t=t)

        _, _, z_c, z_h, z_w = z.shape
        z = self.position_embedding(z)

        # reshape for transformer
        z = E.rearrange(z, "b t c h w -> b (t h w) c")

        z = self.transformerA(z)
        z = E.rearrange(z, "b (t h w) c -> b t h w c", t=t,
                        h=z_h, w=z_w)   # expand for pooling

        # Sum Pooling then scale to get K slots
        z = E.reduce(z, "b t (i h) (j w) c -> b t i j c", "sum", h=2, w=2)
        z = z / 2  # 16 / (8*8) = 1/2

        # reshape for second transformer
        z = E.rearrange(z, "b t i j c -> b (t i j) c", t=t)

        z = self.transformerB(z)
        z = E.rearrange(z, "b (t k) c -> b k t c", t=t,
                        k=self.num_slots)  # NOTE: reverse K<->T

        obj_mean = z.mean(dim=2)  # object mean is taken across frames
        frame_mean = z.mean(dim=1)  # frame mean is taken across objects

        obj_params = self.lambda_obj_mlp(obj_mean)
        frame_params = self.lambda_frame_mlp(frame_mean)

        return obj_params, frame_params


class SIMONE(nn.Module):
    def __init__(
        self,
        input_size,
        K_slots=16,
        z_dim=64,
        recon_alpha=0.2,
        obj_kl_beta=1e-4,
        frame_kl_beta=1e-4,
        pixel_std=0.08,
        device="cuda"
    ):
        """
        SIMONe model
        Args:
            input_size: tuple of (b, t, c, h, w)
            hidden_dim: channels in conv layers
        """

        super(SIMONE, self).__init__()

        b, t, c, h, w = input_size

        self.K_slots = K_slots
        self.z_dim = z_dim
        self.encoder = Encoder(input_size, z_dim=z_dim)
        self.decoder = networks.ConvNet(num_layers=4, in_channels=h + 3, mid_channels=128, out_channels=4,
                                        kernel=1, stride=1, padding=0, dim=2)
        self.layer_norm = nn.LayerNorm((t, h, w, K_slots, 1))

        self.recon_alpha = recon_alpha
        self.obj_kl_beta = obj_kl_beta
        self.frame_kl_beta = frame_kl_beta
        self.pixel_std = torch.tensor(pixel_std).to(device)

        self.device = device

    def encode(self, x):
        """ Encodes sequence of images into frame and object latents

        Params:
        -------
            x: torch.Tensor:
                b, t, c, h, w sized tensor of input images
        Returns:
        --------
            frame_means, object_means:
                Means of object and frame latents
        """
        b, t, c, h, w = x.shape
        obj_params, frame_params = self.encoder(x)

        # sample from latents
        obj_means, obj_stds = torch.split(obj_params, self.z_dim, -1)
        frame_means, frame_stds = torch.split(frame_params, self.z_dim, -1)

        obj_dist = D.Independent(D.Normal(
            obj_means, torch.exp(obj_stds)), 1)
        frame_dist = D.Independent(D.Normal(
            frame_means, torch.exp(frame_stds)), 1)

        return obj_dist, frame_dist

    def decode(self, obj_posterior, frame_posterior, batch_size):
        """ Decodes pixel means and Gaussian mixture logits given
            object and frame means and log vars. Samples independent
            object latents for each pixel across (batch, time, height, width)
            and passes them through 1x1 convnet

        Parameters:
        -----------
        obj_posterior: torch.Tensor
            (B, K, latent_dim * 2) tensor containing means and logvars for object latents
        frame_posterior: torch.Tensor
            (B, T, latent_dim * 2) tensor containing means and logvars for frame latents
        batch_size: tuple
            tuple containing batch size info
        """
        b, t, c, h, w = batch_size

        # Sample latents
        obj_latents = obj_posterior.rsample((t, h, w))
        frame_latents = frame_posterior.rsample((self.K_slots, h, w))

        # Keep channels last for ease of concatenating inputs
        obj_latents = E.rearrange(obj_latents, "t h w b k c -> b k t h w c")
        frame_latents = E.rearrange(
            frame_latents, "k h w b t c -> b k t h w c")

        # Construct spatial coordinate map
        xs = torch.linspace(-1, 1, h)
        ys = torch.linspace(-1, 1, w)
        xb, yb = torch.meshgrid(xs, ys, indexing="ij")
        _coord_map = torch.stack([xb, yb])
        spatial_coords = E.repeat(
            _coord_map, "c h w -> b k t h w c", b=b, k=self.K_slots, t=t
        )

        # Construct temporal map
        temporal_coords = E.repeat(torch.arange(
            0, t), "t -> b k t h w 1", b=b, k=self.K_slots, h=h, w=w)

        # inputs consist of concatenated object and frame latents as well as
        # a spatial coordinate map (ie; broadcast decoder) and temporal coordinates
        latents = torch.cat([obj_latents, frame_latents,
                             spatial_coords, temporal_coords], axis=-1)
        latents = E.rearrange(latents, "b k t h w c -> (b k t) c h w")

        x_recon_masks = self.decoder(latents)
        x_recon = torch.sigmoid(x_recon_masks[:, :3, :])
        x_masks = F.softmax(x_recon_masks[:, 3:, :], dim=1)
        x_recon = E.rearrange(
            x_recon, "(b k t) c h w -> b k t c h w", b=b, k=self.K_slots, t=t)
        x_masks = E.rearrange(
            x_masks, "(b k t) c h w -> b k t c h w", b=b, k=self.K_slots, t=t)

        return x_recon, x_masks

    def forward(self, x, decode_idxs=None):
        b, t, c, h, w = x.shape
        obj_posterior, frame_posterior = self.encode(x)

        x_recon, x_masks = self.decode(
            obj_posterior, frame_posterior, (b, t, c, h, w))

        # Mixture over masks and pixel reconstructions
        x_recon_full = torch.sum(x_recon * x_masks, axis=1)  # sum over slots
        print("recon: ", x_recon.shape)
        print("masks: ", x_masks.shape)
        print("recon full: ", x_recon_full.shape)

        # Compute Negative log likelihood of slot distributions given image
        p_x = D.Independent(D.Normal(
            x_recon_full, torch.ones_like(x_recon_full) * self.pixel_std), 1)
        nll = -p_x.log_prob(x).mean()

        # KL Divergence for frame and object priors against normal gaussian (N(0, 1))
        obj_prior = D.Independent(D.Normal(
            torch.zeros_like(obj_posterior.base_dist.loc),
            torch.ones_like(obj_posterior.base_dist.scale),
        ), 1)
        object_kl = D.kl.kl_divergence(
            obj_posterior, obj_prior).mean()  # Mean over b+t

        frame_prior = D.Independent(D.Normal(
            torch.zeros_like(frame_posterior.base_dist.loc),
            torch.ones_like(frame_posterior.base_dist.scale),
        ), 1)
        frame_kl = D.kl.kl_divergence(
            frame_posterior, frame_prior).mean()  # Mean over b+t

        loss = self.recon_alpha * nll + self.obj_kl_beta * \
            object_kl + self.frame_kl_beta * frame_kl

        return {
            "loss/total": loss,
            "loss/nll": nll,
            "loss/frame_kl": frame_kl,
            "loss/obj_kl": object_kl,
            "recon": x_recon,
            "masks": x_masks,
            "recon_full": x_recon_full
        }

    def kld(self, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


if __name__ == "__main__":
    device = "cpu"
    img = torch.rand(1, 10, 3, 128, 128).to(device)
    model = SIMONE(img.shape, 128, device=device).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
    for i in range(1000):
        optim.zero_grad()
        out = model(img)
        losses = model.compute_loss(img, out)
        losses["total_loss"].backward()
        print(losses["total_loss"].item())
        optim.step()
