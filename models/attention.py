import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def scaled_dot_product(q, k, v, mask=None):
    """
    :param q: (torch.tensor) queries sized (B, K, D)
    :param k: (torch.tensor) keys sized (B, K, D)
    :param v: (torch.tensor) values sized (B, D, V)
    :param mask: (torch.tensor) indexes to mask
    """
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / np.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dim_head=None):
        """
        implements MultiHead Self-Attention
        :param embed_dim: (int) dimension of token embedding
        :param num_heads: (int) how many heads to use
        :param dim_head: (int) head dimension (if None, will be dim / heads)
        """
        super(MultiHeadAttention, self).__init__()
        self.dim_head = (
            embed_dim // num_heads) if dim_head is None else dim_head
        self.heads = num_heads
        dim = self.dim_head * num_heads
        self.to_qvk = nn.Linear(embed_dim, dim * 3, bias=False)
        self.W_0 = nn.Linear(dim, embed_dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None, return_attention=False):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # (batch, tokens, dim*3*heads)

        # (3, batch, heads, tokens, dim_head)
        q, k, v = tuple(
            rearrange(qkv, "bt(hdn)->nbhtd ", n=3, h=self.heads))

        # (batch, heads, tokens, tokens)
        scaled_dot_prod = (
            torch.einsum("bhid,bhjd->bhij", q, k) *
            self.scale_factor
        )

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attention, v)
        out = rearrange(out, "bhtd->bt(hd)")

        return self.W_0(out)


class SlotAttention(nn.Module):
    def __init__(
        self, num_iterations, num_slots, slot_dim, mlp_hidden_dim, epsilon=1e-8
    ):
        """
        Slot attention
        :param num_iterations: int: how many iterations to pay attention over
        :param num_slots: int: how many slots to initialize
        :param slot_dim: int: dimensions of each slot
        :param mlp_hidden_dim: int: size of hidden layer in mlp
        """
        super(SlotAttention, self).__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.epsilon = epsilon

        self.queries = nn.Linear(slot_dim, slot_dim, bias=False)
        self.keys = nn.Linear(slot_dim, slot_dim, bias=False)
        self.values = nn.Linear(slot_dim, slot_dim, bias=False)

        self.input_norm = nn.LayerNorm(slot_dim)
        self.mlp_norm = nn.LayerNorm(slot_dim)
        self.slot_norm = nn.LayerNorm(slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim),
        )

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.slots_mu = torch.empty((1, 1, slot_dim))
        self.slots_log_sigma = torch.empty((1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)

    def forward(self, x, slot_initialization=None):
        b, d = x.shape[0], x.shape[-1]

        x = self.input_norm(x)
        keys = self.keys(x)  # [batch_size, input_size, slot_dim]
        values = self.values(x)  # [batch_size, input_size, slot_dim]

        mu = self.slots_mu.expand(b, self.num_slots, -1).to(device)
        sigma = self.slots_log_sigma.exp().expand(b, self.num_slots, -1).to(device)

        if slot_initialization is not None:
            slots = slot_initialization
        else:
            slots = mu + sigma * torch.randn(
                mu.shape, device=device
            )  # [batch_size, num_slots, slot_dim]

        slots = slots.to(device)
        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.slot_norm(slots)  # [batch_size, num_slots, slot_dim]

            # Compute attention
            queries = self.queries(slots)  # [batch_size, num_slots, slot_dim]

            # b = batch_size, i = input_size, d = slot_dim, k = num_slots
            attn_logits = torch.einsum(
                "bid,bkd->bik", queries, keys
            )  # [batch_size, input_size, num_slots]
            attn_logits *= self.slot_dim ** -0.5
            attn = F.softmax(attn_logits, dim=1)

            # Weighted Mean
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-1, keepdim=True)

            updates = torch.einsum(
                "bkd,bik->bid", values, attn
            )  # [batch_size, num_slots, slot_dim]

            updates = updates.reshape(-1, d)
            slots_prev = slots_prev.reshape(-1, d)

            slots = self.gru(updates, slots_prev)

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.mlp_norm(slots))

        return slots
