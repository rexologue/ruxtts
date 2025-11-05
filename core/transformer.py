from typing import TypeVar, overload, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from .generic_utils import exists

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

_T = TypeVar("_T")

@overload
def cast_tuple(val: _T, depth: int = 1) -> tuple[_T]: ...
@overload
def cast_tuple(val: list[_T], depth: int = 1) -> tuple[_T]: ...
@overload
def cast_tuple(val: tuple[_T, ...], depth: int = 1) -> tuple[_T]: ...

def cast_tuple(val, depth: int = 1):
    """
    Canonicalize input into a tuple.

    - If `val` is a list, convert to tuple.
    - If `val` is already a tuple, return as-is.
    - Otherwise, replicate the scalar `val` `depth` times into a tuple.
    """
    if isinstance(val, list):
        return tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth


def max_neg_value(t: torch.Tensor) -> torch.Tensor:
    """
    Return a large negative value suitable for masking logits.
    Uses finfo for the tensor dtype (float dtypes expected).
    """
    return -torch.finfo(t.dtype).max # type: ignore


def stable_softmax(t: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """
    Numerically stable softmax with optional temperature.
    The scaling & max-subtraction are standard tricks to avoid overflow.

    NOTE:
    - We DO NOT re-multiply by 'temperature' after subtracting the max.
      That would distort the distribution (acts like inverse temperature).
    """
    t = t / temperature
    t = t - torch.amax(t, dim=dim, keepdim=True)
    return t.softmax(dim=dim)


def route_args(router: dict[str, Iterable[tuple[bool, bool]]], args: dict, depth: int):
    """
    Route keyword arguments to a sequence of (f, g) layer pairs.

    router[key] is expected to be an iterable of length == depth, where each element is (route_to_f: bool, route_to_g: bool).
    For each depth level, we construct two dicts: f_args and g_args, merging routed keys.
    """
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        # 'd' is the depth index. Avoid overshadowing the function arg 'depth'.
        for d, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = (({key: val} if route else {}) for route in routes)
            routed_args[d] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args


# ---------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------

class SequentialSequence(nn.Module):
    """
    Execute a sequence of (f, g) residual sublayers:
        x = x + f(x, **f_args)
        x = x + g(x, **g_args)

    Optionally supports stochastic depth via 'layer_dropout' (applied per sublayer independently).
    """
    def __init__(self, layers: nn.ModuleList, args_route: dict = {}, layer_dropout: float = 0.0):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), (
            "each argument route map must have the same depth as the number of sequential layers"
        )
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = float(layer_dropout)

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))
        do_drop = self.training and self.layer_dropout > 0.0

        for (f, g), (f_args, g_args) in layers_and_args: # type: ignore
            # stochastic depth for f
            if not (do_drop and torch.rand((), device=x.device) < self.layer_dropout):
                x = x + f(x, **f_args)
            # stochastic depth for g
            if not (do_drop and torch.rand((), device=x.device) < self.layer_dropout):
                x = x + g(x, **g_args)
        return x


class DivideMax(nn.Module):
    """
    Divide tensor by its per-dim maximum (detached), with epsilon to avoid division by zero.
    """
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maxes = x.amax(dim=self.dim, keepdim=True).detach()
        return x / (maxes + self.eps)


# https://arxiv.org/abs/2103.17239
class LayerScale(nn.Module):
    """
    Scales the output of a sub-layer by a small, learnable factor depending on depth.
    Residual addition is handled by the caller.
    """
    def __init__(self, dim: int, depth: int, fn: nn.Module):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        # Parameter of shape (1, 1, dim) to broadcast over (B, N, D)
        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    """
    Pre-LayerNorm wrapper with optional 'sandwich' normalization (post-norm),
    i.e., Norm -> Fn -> (Norm or Identity)
    """
    def __init__(self, dim: int, fn: nn.Module, sandwich: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)


class GEGLU(nn.Module):
    """
    Gated GELU unit:
        split last dim in half -> (x, gates)
        output = x * GELU(gates)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    """
    Standard Transformer FFN with GEGLU activation.
    """
    def __init__(self, dim: int, dropout: float = 0.0, mult: float = 4.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner * 2),  # doubled for GEGLU
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------

class Attention(nn.Module):
    """
    Multi-Head Self-Attention with:
    - numerically stable softmax
    - robust causal masking (works for i != j)
    - optional external key padding mask of shape (B, N_k)
    - lightweight cache of a causal mask per (i, j) on device
    """
    def __init__(
        self,
        dim: int,
        seq_len: Optional[int],            # kept for API parity; not strictly required here
        causal: bool = True,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5
        self.temperature = float(temperature)

        self.causal = bool(causal)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # Lightweight cache: last causal mask (i, j, dtype=bool) for the current device
        self.register_buffer("_cached_causal_mask", None, persistent=False)
        self._cached_shape: tuple[int, int] | None = None

    @staticmethod
    def _build_causal_mask(i: int, j: int, device: torch.device) -> torch.Tensor:
        """
        Build a boolean mask with True where future positions should be masked.
        Works for non-square (i != j) attention matrices.
        """
        qi = torch.arange(i, device=device).unsqueeze(-1)  # (i, 1)
        kj = torch.arange(j, device=device).unsqueeze(0)   # (1, j)
        # True where key index > query index (future positions)
        return kj > qi

    def _get_causal_mask(self, i: int, j: int, device: torch.device) -> torch.Tensor:
        """
        Return a cached causal mask if shape/device matches; otherwise (re)build.
        """
        if (self._cached_causal_mask is None) or (self._cached_shape != (i, j)) or (self._cached_causal_mask.device != device):
            self._cached_causal_mask = self._build_causal_mask(i, j, device)
            self._cached_shape = (i, j)
        return self._cached_causal_mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (B, N, D_model)
        mask: optional key padding mask of shape (B, N_k) with True for valid tokens.
              If provided, it will be broadcast to (B, 1, 1, N_k).
        """
        b, n, d = x.shape
        h = self.heads
        device = x.device

        # Project to queries, keys, values and split heads: (B, H, N, D_h)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v))

        # Query scaling for dot-product stability
        q = q * self.scale

        # Attention logits: (B, H, N_q, N_k)
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        mask_value = max_neg_value(dots)

        # Optional key padding mask (B, N_k) -> (B, 1, 1, N_k)
        if exists(mask):
            key_mask = rearrange(mask, "b j -> b () () j")
            dots.masked_fill_(~key_mask, mask_value)  # type: ignore
            del key_mask

        # Causal mask (future positions): robust for i != j
        if self.causal:
            i, j = dots.shape[-2:]
            causal_mask = self._get_causal_mask(i, j, device)
            dots.masked_fill_(causal_mask, mask_value)

        # Stable softmax (with optional temperature)
        attn = stable_softmax(dots, dim=-1, temperature=self.temperature)

        # Weighted sum of values -> merge heads
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


# ---------------------------------------------------------------------
# Main Transformer
# ---------------------------------------------------------------------

class Transformer(nn.Module):
    """
    A stack of attention + feedforward blocks, each wrapped with:
      - PreNorm
      - LayerScale
      - Residual applied by SequentialSequence

    Notes:
    - 'sparse_attn' is accepted for API parity, but not used in this implementation.
      If you plan to add sparse layers, route and instantiate them here.
    - 'sandwich_norm' applies an extra LayerNorm after each sublayer (PreNorm(..., sandwich=True)).
    """
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        seq_len: Optional[int],
        causal: bool = True,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: float = 4.0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        sparse_attn: bool | tuple[bool, ...] = False,   # accepted but unused
        sandwich_norm: bool = False,
        layer_dropout: float = 0.0,                     # stochastic depth prob per sublayer
        attn_temperature: float = 1.0,                  # pass-through to Attention
    ):
        super().__init__()
        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)   # kept for API compatibility

        for ind, _sparse_flag in zip(range(depth), sparse_layer):
            attn = Attention(
                dim=dim,
                causal=causal,
                seq_len=seq_len,
                heads=heads,
                dim_head=dim_head,
                dropout=attn_dropout,
                temperature=attn_temperature,
            )

            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

            layers.append(
                nn.ModuleList(
                    [
                        LayerScale(dim, ind + 1, PreNorm(dim, attn, sandwich=sandwich_norm)),
                        LayerScale(dim, ind + 1, PreNorm(dim, ff, sandwich=sandwich_norm)),
                    ]
                )
            )

        execute_type = SequentialSequence
        route_attn = ((True, False),) * depth  # route 'mask' only to attention (f), not to feed-forward (g)
        attn_route_map = {"mask": route_attn}

        self.layers = execute_type(layers, args_route=attn_route_map, layer_dropout=layer_dropout)

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Pass-through kwargs (e.g., mask=...) will be routed to attention only,
        as defined in attn_route_map.
        """
        return self.layers(x, **kwargs)
