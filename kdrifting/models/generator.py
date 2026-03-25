"""DiT-style generator ported from the JAX implementation."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from kdrifting.models.common import RMSNorm


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Create a 1-D sinusoidal position embedding from grid coordinates."""
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Create a flattened 2-D sinusoidal position embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    stacked = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    half = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(half, stacked[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(half, stacked[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply AdaLN modulation to token activations."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rope(
    q: Tensor,
    k: Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Apply rotary positional embeddings to attention queries and keys."""
    _, seq_len, _, head_dim = q.shape
    half_dim = head_dim // 2
    freq_range = torch.arange(0, half_dim, device=q.device, dtype=dtype) / half_dim
    freqs = (1.0 / (10000**freq_range)).to(device=q.device)
    t = torch.arange(seq_len, device=q.device, dtype=dtype)
    freqs = torch.outer(t, freqs)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = torch.cos(emb).unsqueeze(0).unsqueeze(2)
    sin = torch.sin(emb).unsqueeze(0).unsqueeze(2)

    def rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _init_linear(module: nn.Linear, *, weight_init: str, bias_init: str) -> None:
    if weight_init == "xavier_uniform":
        nn.init.xavier_uniform_(module.weight)
    elif weight_init == "zeros":
        nn.init.zeros_(module.weight)
    elif weight_init == "normal":
        nn.init.normal_(module.weight, std=0.02)
    else:
        raise ValueError(f"Unsupported weight init: {weight_init}")

    bias = cast(Tensor | None, getattr(module, "bias", None))
    if bias is None:
        return
    if bias_init == "zeros":
        nn.init.zeros_(bias)
    else:
        nn.init.constant_(bias, 0.0)


class TorchLinear(nn.Module):
    """Linear layer wrapper with source-compatible initialization."""

    def __init__(
        self,
        features_in: int,
        features_out: int,
        *,
        bias: bool = True,
        weight_init: str = "xavier_uniform",
        bias_init: str = "zeros",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(features_in, features_out, bias=bias)
        _init_linear(self.linear, weight_init=weight_init, bias_init=bias_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = TorchLinear(hidden_size, intermediate_size)
        self.w3 = TorchLinear(hidden_size, intermediate_size)
        self.proj = TorchLinear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(functional.silu(self.w1(x)) * self.w3(x))


class StandardMLP(nn.Module):
    """Standard GELU MLP block used when SwiGLU is disabled."""

    def __init__(self, hidden_size: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = TorchLinear(hidden_size, mlp_hidden_dim)
        self.fc2 = TorchLinear(mlp_hidden_dim, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(functional.gelu(self.fc1(x), approximate="none"))


class Attention(nn.Module):
    """Multi-head attention block with optional RMSNorm and RoPE."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        use_rope: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.use_rope = use_rope
        self.attn_fp32 = attn_fp32
        self.qkv = TorchLinear(dim, dim * 3, bias=qkv_bias)
        self.proj = TorchLinear(dim, dim, bias=True)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)
        head_dim = dim // num_heads
        if qk_norm:
            if use_rmsnorm:
                self.q_norm: nn.Module = RMSNorm(head_dim)
                self.k_norm: nn.Module = RMSNorm(head_dim)
            else:
                self.q_norm = nn.LayerNorm(head_dim, eps=1e-6, elementwise_affine=True)
                self.k_norm = nn.LayerNorm(head_dim, eps=1e-6, elementwise_affine=True)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x: Tensor,
        *,
        deterministic: bool = True,
        return_qk: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        batch, seq_len, channels = x.shape
        head_dim = self.dim // self.num_heads
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, head_dim)
        q = self.q_norm(qkv[:, :, 0])
        k = self.k_norm(qkv[:, :, 1])
        v = qkv[:, :, 2]

        if self.use_rope:
            rope_dtype = torch.float32 if self.attn_fp32 else q.dtype
            q, k = apply_rope(q, k, dtype=rope_dtype)

        qk = (q, k) if return_qk else None

        if self.attn_fp32:
            q = q.to(dtype=torch.float32) * (head_dim**-0.5)
            k = k.to(dtype=torch.float32)
            v = v.to(dtype=torch.float32)
        else:
            q = q * (head_dim**-0.5)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = functional.softmax(attn, dim=-1)
        if self.attn_dropout.p > 0:
            attn = self.attn_dropout(attn) if not deterministic else attn

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, channels)
        out = self.proj(out)
        if self.proj_dropout.p > 0:
            out = self.proj_dropout(out) if not deterministic else out
        return out, qk


class LightningDiTBlock(nn.Module):
    """Main DiT transformer block with AdaLN conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
        use_rope: bool = False,
        attn_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = (
            RMSNorm(hidden_size)
            if use_rmsnorm
            else nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        )
        self.norm2 = (
            RMSNorm(hidden_size)
            if use_rmsnorm
            else nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        )
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_rope=use_rope,
            attn_fp32=attn_fp32,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            hid_size = int((2.0 / 3.0) * mlp_hidden_dim)
            hid_size = ((hid_size + 31) // 32) * 32
            self.mlp: nn.Module = SwiGLUFFN(hidden_size, hid_size)
        else:
            self.mlp = StandardMLP(hidden_size, mlp_hidden_dim)

        self.adaln = nn.Sequential(
            nn.SiLU(),
            TorchLinear(hidden_size, 6 * hidden_size, weight_init="zeros", bias_init="zeros"),
        )

    def forward(self, x: Tensor, c: Tensor, *, deterministic: bool = True) -> Tensor:
        shifts = self.adaln(c.to(dtype=torch.float32)).to(dtype=x.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = shifts.chunk(6, dim=1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm, deterministic=deterministic)[0]

        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
    """Final AdaLN-conditioned projection layer."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        *,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        self.norm = (
            RMSNorm(hidden_size)
            if use_rmsnorm
            else nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        )
        self.adaln = nn.Sequential(
            nn.SiLU(),
            TorchLinear(hidden_size, 2 * hidden_size, weight_init="zeros", bias_init="zeros"),
        )
        self.proj = TorchLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            weight_init="zeros",
            bias_init="zeros",
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaln(c.to(dtype=torch.float32)).to(dtype=x.dtype).chunk(2, dim=1)
        return self.proj(modulate(self.norm(x), shift, scale))


class LightningDiT(nn.Module):
    """Transformer backbone used by the generator."""

    def __init__(
        self,
        *,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_channels: int = 32,
        cond_dim: int | None = None,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        n_cls_tokens: int = 0,
        attn_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.out_channels = out_channels
        self.n_cls_tokens = n_cls_tokens
        self.cond_dim = hidden_size if cond_dim is None else cond_dim
        patch_dim = patch_size * patch_size * in_channels
        self.patch_embed = TorchLinear(patch_dim, self.hidden_size)
        self.cls_proj = TorchLinear(self.cond_dim, hidden_size) if n_cls_tokens > 0 else None
        self.blocks = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    use_rope=use_rope,
                    attn_fp32=attn_fp32,
                )
                for _ in range(depth)
            ],
        )
        self.final_layer = FinalLayer(
            hidden_size,
            patch_size,
            out_channels,
            use_rmsnorm=use_rmsnorm,
        )
        self.cls_embed = (
            nn.Parameter(torch.empty(1, n_cls_tokens, hidden_size)) if n_cls_tokens > 0 else None
        )
        if self.cls_embed is not None:
            nn.init.normal_(self.cls_embed, std=0.02)
        num_patches = (input_size // patch_size) ** 2
        grid_size = int(math.sqrt(num_patches))
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size)
        pos_tensor = torch.as_tensor(pos_embed, dtype=torch.float32).unsqueeze(0)
        self.pos_embed = nn.Parameter(pos_tensor)

    def forward(self, x: Tensor, c: Tensor, *, deterministic: bool = True) -> Tensor:
        batch, height, _, channels = x.shape
        target_grid = self.input_size // self.patch_size
        num_patches = target_grid * target_grid
        effective_patch = height // target_grid
        grid_h, grid_w = target_grid, target_grid

        x = x.reshape(
            batch,
            grid_h,
            effective_patch,
            grid_w,
            effective_patch,
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            batch,
            num_patches,
            effective_patch * effective_patch * channels,
        )
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)

        if self.n_cls_tokens > 0:
            assert self.cls_proj is not None
            assert self.cls_embed is not None
            c_tokens = self.cls_proj(c).unsqueeze(1).repeat(1, self.n_cls_tokens, 1)
            c_tokens = c_tokens + self.cls_embed.to(device=x.device, dtype=x.dtype)
            x = torch.cat([c_tokens, x], dim=1)

        for block in self.blocks:
            x = block(x, c, deterministic=deterministic)

        x = self.final_layer(x, c)
        if self.n_cls_tokens > 0:
            x = x[:, self.n_cls_tokens :, :]

        x = x.reshape(
            batch,
            grid_h,
            grid_w,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            batch,
            self.input_size,
            self.input_size,
            self.out_channels,
        )
        return x


class TimestepEmbedder(nn.Module):
    """Embed scalar CFG values using sinusoidal features plus an MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.fc1 = TorchLinear(frequency_embedding_size, hidden_size, weight_init="normal")
        self.fc2 = TorchLinear(hidden_size, hidden_size, weight_init="normal")

    def forward(self, t: Tensor) -> Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half,
        )
        args = t[:, None].to(dtype=torch.float32) * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.fc2(functional.silu(self.fc1(embedding)))


class DitGen(nn.Module):
    """Top-level generator module that samples noise and class conditioning."""

    def __init__(
        self,
        *,
        cond_dim: int,
        num_classes: int = 1001,
        noise_classes: int = 0,
        noise_coords: int = 1,
        input_size: int = 32,
        in_channels: int = 3,
        n_cls_tokens: int = 0,
        patch_size: int = 2,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_channels: int = 3,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        use_bf16: bool = False,
        attn_fp32: bool = True,
        use_remat: bool = False,
    ) -> None:
        del use_remat
        super().__init__()
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.noise_classes = noise_classes
        self.noise_coords = noise_coords
        self.input_size = input_size
        self.in_channels = in_channels
        self.use_bf16 = use_bf16
        self.class_embed = nn.Embedding(num_classes, cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        self.noise_embeds = nn.ModuleList(
            [nn.Embedding(noise_classes, cond_dim) for _ in range(noise_coords)]
            if noise_classes > 0
            else [],
        )
        for embed in self.noise_embeds.children():
            typed_embed = cast(nn.Embedding, embed)
            nn.init.normal_(typed_embed.weight, std=0.02)

        self.cfg_embedder = TimestepEmbedder(cond_dim)
        self.cfg_norm = RMSNorm(cond_dim)
        self.model = LightningDiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_channels=out_channels,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
            n_cls_tokens=n_cls_tokens,
            attn_fp32=attn_fp32,
            cond_dim=cond_dim,
        )

    def dummy_input(self) -> dict[str, Any]:
        return {
            "c": torch.ones(1, dtype=torch.int64),
            "cfg_scale": 1.0,
            "temp": 1.0,
            "deterministic": True,
        }

    def c_cfg_noise_to_cond(
        self,
        c: Tensor,
        cfg_scale: float | Tensor,
        noise_labels: Tensor,
    ) -> Tensor:
        cond = self.class_embed(c)
        if self.noise_classes > 0:
            for index, embed in enumerate(self.noise_embeds):
                cond = cond + embed(noise_labels[:, index])

        if isinstance(cfg_scale, Tensor):
            cfg_scale_t = cfg_scale.to(device=c.device, dtype=torch.float32)
            if cfg_scale_t.ndim == 0:
                cfg_scale_t = cfg_scale_t.repeat(c.shape[0])
        else:
            cfg_scale_t = torch.full(
                (c.shape[0],),
                float(cfg_scale),
                device=c.device,
                dtype=torch.float32,
            )
        cond = cond + self.cfg_norm(self.cfg_embedder(cfg_scale_t)) * 0.02
        if self.use_bf16:
            cond = cond.to(dtype=torch.bfloat16)
        return cond

    def generate_image(self, x: Tensor, cond: Tensor, *, deterministic: bool = True) -> Tensor:
        return self.model(x, cond, deterministic=deterministic)

    def sample_noise(
        self,
        *,
        batch_size: int,
        device: torch.device,
        temp: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample latent noise and auxiliary noise labels for generation."""
        x = torch.randn(
            batch_size,
            self.input_size,
            self.input_size,
            self.in_channels,
            device=device,
            generator=generator,
        )
        x = x * temp
        if self.use_bf16:
            x = x.to(dtype=torch.bfloat16)

        noise_labels = torch.randint(
            0,
            max(1, self.noise_classes),
            (batch_size, max(1, self.noise_coords)),
            device=device,
            generator=generator,
        )
        return x, noise_labels

    def forward(
        self,
        c: Tensor,
        cfg_scale: float | Tensor = 1.0,
        temp: float = 1.0,
        deterministic: bool = True,
        train: bool = False,
        generator: torch.Generator | None = None,
        noise_x: Tensor | None = None,
        noise_labels: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        del train
        batch = c.shape[0]
        has_noise_x = noise_x is not None
        has_noise_labels = noise_labels is not None
        if has_noise_x != has_noise_labels:
            raise ValueError("Expected both noise_x and noise_labels when overriding noise.")

        if noise_x is None:
            x, sampled_noise_labels = self.sample_noise(
                batch_size=batch,
                device=c.device,
                temp=temp,
                generator=generator,
            )
        else:
            x = noise_x.to(device=c.device)
            sampled_noise_labels = cast(Tensor, noise_labels).to(device=c.device, dtype=torch.int64)
            expected_x_shape = (batch, self.input_size, self.input_size, self.in_channels)
            if tuple(x.shape) != expected_x_shape:
                raise ValueError(
                    f"noise_x shape mismatch: expected {expected_x_shape}, got {tuple(x.shape)}",
                )
            expected_noise_shape = (batch, max(1, self.noise_coords))
            if tuple(sampled_noise_labels.shape) != expected_noise_shape:
                raise ValueError(
                    "noise_labels shape mismatch: "
                    f"expected {expected_noise_shape}, got {tuple(sampled_noise_labels.shape)}",
                )
            if self.use_bf16:
                x = x.to(dtype=torch.bfloat16)

        cond = self.c_cfg_noise_to_cond(c, cfg_scale, sampled_noise_labels)
        samples = self.generate_image(x, cond, deterministic=deterministic)
        return {
            "samples": samples,
            "noise": {
                "x": x,
                "noise_labels": sampled_noise_labels,
            },
        }


def build_generator_from_config(model_config: dict[str, Any]) -> DitGen:
    """Build a generator from a full configuration dictionary."""
    return DitGen(**dict(model_config))
