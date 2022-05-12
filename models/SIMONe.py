import os
import torch
import einops as E
import networks
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def MultivariateNormal(loc, scale):
    return torch.distributions.independent.Independent(
        torch.distributions.normal.Normal(loc, scale), 1
    )


class Encoder(nn.Module):
    """
    Outputs frame encoding for each frame in a batch (B, T, H, W, C)
    Frame latents are given 3D position encodings to maintain
    temporal order before getting sent through transformer blocks.
    """

    def __init__(self, input_size, num_slots=16, z_dim=32, feature_dim=64):
        super(Encoder, self).__init__()
        b, t, c, h, w = input_size
        self.num_slots = num_slots
        self.z_dim = z_dim

        # Number of layers in conv net is dependent on resolution
        num_layers = np.log2(w / 8)
        self.ConvBlock = networks.ConvNet(
            num_layers,
            in_channels=c,
            mid_channels=feature_dim,
            out_channels=feature_dim,
        )

        self.position_encoding_1 = networks.PositionalEncodingPermute3D(128)
        self.position_encoding_2 = networks.PositionalEncodingPermute3D(128)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=4, dim_feedforward=1024, dropout=0.0,
        )
        self.transformerA = nn.TransformerEncoder(transformer_encoder_layer, 4)
        self.transformerB = nn.TransformerEncoder(transformer_encoder_layer, 4)

        # MLP from objects --> frames
        self.lambda_frame_mlp = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)
        )

        # MLP from frames --> objects
        self.lambda_obj_mlp = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        assert h == w, f"Height and Width should be equal for input shape: {x.shape}"

        # flatten input along time dim
        x = E.rearrange(x, "b t c h w -> (b t) c h w")

        # output of conv encoder should be b, t, 128, 8, 8
        z = self.ConvBlock(x)
        z = E.rearrange(z, "(b t) c h w -> b t h w c", t=t)

        _, _, z_h, z_w, z_c = z.shape
        z = self.position_encoding_1(z)

        # reshape for transformer
        z = E.rearrange(z, "b t z_h z_w z_c -> b (t z_h z_w) z_c")

        z = self.transformerA(z)
        z = E.rearrange(
            z, "b (t z_h z_w) z_c -> (b t) z_c z_h z_w", t=t, z_h=z_h, z_w=z_w
        )  # expand for pooling

        z = F.avg_pool2d(z, kernel_size=2) * 2

        # Sum Pooling then scale to get K slots
        # z = E.reduce(z, "b t (i h) (j w) c -> b t i j c", "sum", h=2, w=2)
        # z = z / 2  # 16 / (8*8) = 1/2

        # reshape for second transformer
        z = E.rearrange(z, "(b t) z_c z_h z_w -> b t z_h z_w z_c", t=t)
        z = self.position_encoding_2(z)
        z = E.rearrange(z, "b t z_h z_w z_c -> b (t z_h z_w) z_c")

        z = self.transformerB(z)
        z = E.rearrange(
            z, "b (t k) z_c -> b k t z_c", t=t, k=self.num_slots
        )  # NOTE: reverse K<->T

        obj_mean = z.mean(dim=1)  # object mean is taken across frames
        frame_mean = z.mean(dim=2)  # frame mean is taken across objects

        obj_params = self.lambda_obj_mlp(obj_mean)
        frame_params = self.lambda_frame_mlp(frame_mean)

        return obj_params, frame_params


class SIMONE(nn.Module):
    def __init__(
        self,
        input_size,
        K_slots=16,
        recon_alpha=0.2,
        obj_kl_beta=1e-5,
        frame_kl_beta=1e-4,
        pixel_std=0.08,
        device="cuda",
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
        self.decoder = networks.MLP(
            in_features=self.z_dim * 2 + 3,
            n_layers=3,
            intermediate_size=256,
            out_features=4,
        )
        self.layer_norm = nn.LayerNorm((t, K_slots, h, w))

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
        obj_latents, frame_latents = self.encoder(
            x
        )  # params are (b, k, c * 2) and (b, t, c * 2)

        obj_params = E.repeat(obj_latents, "b k c -> b t k h w c", t=t, h=h, w=w)
        frame_params = E.repeat(
            frame_latents, "b t c -> b t k h w c", k=self.K_slots, h=h, w=w
        )

        # split distributions into mean and std
        obj_means, obj_stds = torch.split(obj_params, self.z_dim, -1)
        frame_means, frame_stds = torch.split(frame_params, self.z_dim, -1)

        # create normal distributions
        obj_dist = D.Normal(obj_means, torch.exp(obj_stds))
        frame_dist = D.Normal(frame_means, torch.exp(frame_stds))

        return obj_dist, frame_dist, obj_latents, frame_latents

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
        obj_latents = obj_posterior.rsample()  # b t k h w c
        frame_latents = frame_posterior.rsample()  # b t k h w c

        # Keep channels last for ease of concatenating inputs

        # Construct spatial coordinate map
        xs = torch.linspace(-1, 1, h).to(self.device)
        ys = torch.linspace(-1, 1, w).to(self.device)
        xb, yb = torch.meshgrid(xs, ys, indexing="ij")
        _coord_map = torch.stack([xb, yb])
        spatial_coords = E.repeat(
            _coord_map, "c h w -> b t k h w c", b=b, k=self.K_slots, t=t
        )

        # Construct temporal map
        temporal_coords = E.repeat(
            torch.arange(0, t), "t -> b t k h w 1", b=b, k=self.K_slots, h=h, w=w
        )

        # inputs consist of concatenated object and frame latents as well as
        # a spatial coordinate map (ie; broadcast decoder) and temporal coordinates
        latents = torch.cat(
            [obj_latents, frame_latents, spatial_coords, temporal_coords], axis=-1
        )
        latents = E.rearrange(latents, "b t k h w c -> (b t k h w) c")

        x_recon_masks = self.decoder(latents)
        x_recon_masks = E.rearrange(
            x_recon_masks,
            "(b t k h w) c -> b t k h w c",
            t=t,
            k=self.K_slots,
            h=h,
            w=w,
            c=4,
        )

        pixels = x_recon_masks[..., :3]
        weights = x_recon_masks[..., 3]

        weights = F.layer_norm(weights, (t, self.K_slots, h, w))
        weights_softmax = F.softmax(weights, dim=2)

        weighted_pixels = (pixels * weights_softmax.unsqueeze(-1)).sum(dim=2)

        return pixels, weights, weights_softmax, weighted_pixels

    def forward(self, x, decode_idxs=None):
        """
        Returns dictionary containing:
            total loss, negative log likelihood, object and
            frame KL divergences, as well as output masks,
            pixel mixture distributions and the full reconstruction
        """
        b, t, c, h, w = x.shape
        obj_posterior, frame_posterior, obj_latents, frame_latents = self.encode(x)

        pixels, weights, weights_softmax, weighted_pixels = self.decode(
            obj_posterior, frame_posterior, (b, t, c, h, w)
        )

        # Mixture over masks and pixel reconstructions
        pixel_likelihood = self.pixel_likelihood_loss(pixels, x, weights_softmax)
        obj_latent_loss, frame_latent_loss = self.latent_kl_loss(
            obj_latents, frame_latents
        )

        pixel_likelihood = pixel_likelihood.mean()
        obj_latent_loss = obj_latent_loss.mean()
        frame_latent_loss = frame_latent_loss.mean()

        loss = (
            self.recon_alpha * pixel_likelihood
            + self.obj_kl_beta * obj_latent_loss
            + self.frame_kl_beta * frame_latent_loss
        )

        loss = loss
        return {
            "loss/total": loss,
            "loss/nll": pixel_likelihood,
            "loss/frame_kl": frame_latent_loss,
            "loss/obj_kl": obj_latent_loss,
            "weights": weights,
            "recon": weighted_pixels,
            "masks": weights_softmax,
        }

    def pixel_likelihood_loss(self, pixels, video, weights):
        b, t, k, h, w, c = pixels.shape
        assert video.shape == (b, t, c, h, w)
        target = E.repeat(video, "b t c h w -> b t k h w c", k=k)
        log_prob = D.normal.Normal(pixels, self.pixel_std).log_prob(target)

        pixel_probs = torch.exp(log_prob) * weights.unsqueeze(-1)
        pixel_probs = pixel_probs.sum(dim=2)
        assert pixel_probs.shape == (b, t, h, w, c)
        pixel_likelihood = (
            -1 / (t * h * w) * torch.log(pixel_probs).sum(dim=(4, 3, 2, 1))
        )
        assert pixel_likelihood.shape == (b,)
        return pixel_likelihood

    def get_latent_dist(self, latent, log_scale_min=-10, log_scale_max=3):
        """Convert the MLP output (with mean and log std) into a torch `Normal` distribution."""
        means, log_scale = torch.split(latent, self.z_dim, -1)
        # Clamp the minimum to keep latents from getting too far into the saturating region of the exp
        # And the max because I noticed it exploding early in the training sometimes
        log_scale = log_scale.clamp(min=log_scale_min, max=log_scale_max)
        dist = D.normal.Normal(means, torch.exp(log_scale))
        return dist

    def latent_kl_loss(self, obj_latents, frame_latents):

        b, k, c = obj_latents.shape
        b, t, c = frame_latents.shape

        obj_latent_dist = self.get_latent_dist(obj_latents)
        frame_latent_dist = self.get_latent_dist(frame_latents)
        latent_prior = D.normal.Normal(
            torch.zeros(c // 2, device=obj_latents.device, dtype=obj_latents.dtype,),
            scale=1,
        )
        obj_latent_loss = (1 / k) * D.kl.kl_divergence(obj_latent_dist, latent_prior)
        # The KL doesn't reduce all the way because the distribution considers the batch size to be (batch, K, LATENT_CHANNELS)
        obj_latent_loss = obj_latent_loss.sum(dim=(2, 1))
        assert obj_latent_loss.shape == (b,)
        frame_latent_loss = (1 / t) * D.kl.kl_divergence(
            frame_latent_dist, latent_prior
        )
        frame_latent_loss = frame_latent_loss.sum(dim=(2, 1))
        assert frame_latent_loss.shape == (b,)

        return obj_latent_loss, frame_latent_loss


if __name__ == "__main__":
    device = "cpu"
    img = torch.rand(1, 10, 3, 128, 128).to(device)
    model = SIMONE(img.shape, 128, device=device).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
    for i in range(1000):
        optim.zero_grad()
        out = model(img)
        losses = model.compute_loss(img, out)
        losses["loss/total"].backward()
        print(losses["loss/total"].item())
        optim.step()
