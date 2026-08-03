from __future__ import annotations

import os
import sys
import types
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


def _lawam_root() -> Path:
    lawam_root = os.environ.get("LAWAM_ROOT")
    if lawam_root is None:
        lawam_root = str(Path(__file__).resolve().parents[5] / "LaWAM-main")
    return Path(lawam_root)


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_lawam_components():
    """Load LaWAM components without importing Lightning-backed package init files."""
    utils_dir = _lawam_root() / "latent_action_model" / "core" / "utils"
    package_name = "_lawam_lam_utils"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(utils_dir)]
        sys.modules[package_name] = package

    modules_name = f"{package_name}.modules"
    if modules_name not in sys.modules:
        modules_shim = types.ModuleType(modules_name)

        class CategorySpecificLinear(nn.Module):
            def __init__(self, num_categories: int, input_dim: int, hidden_dim: int) -> None:
                super().__init__()
                self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
                self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

            def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
                selected_W = self.W[cat_ids]
                selected_b = self.b[cat_ids]
                return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

        class CategorySpecificMLP(nn.Module):
            def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
                super().__init__()
                self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
                self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

            def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
                squeeze_time = False
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                    squeeze_time = True
                hidden = torch.relu(self.layer1(x, cat_ids))
                out = self.layer2(hidden, cat_ids)
                if squeeze_time:
                    out = out.squeeze(1)
                return out

        modules_shim.CategorySpecificMLP = CategorySpecificMLP

        class CrossAttentionBlock(nn.Module):
            def __init__(self, dim: int, num_heads: int) -> None:
                super().__init__()
                self.norm_q = nn.LayerNorm(dim)
                self.norm_kv = nn.LayerNorm(dim)
                self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

            def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
                out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
                return q + out

        class QFormer_att(nn.Module):
            def __init__(
                self,
                query_dim: int,
                context_dim: int,
                num_frames: int,
                num_queries: int,
                grid_hw: tuple[int, int],
                add_tokens: int = 1,
                num_layers: int = 6,
                num_heads: int = 16,
                ffn_expansion_factor: int = 2,
                dropout: float = 0.1,
                use_mask: bool = False,
            ) -> None:
                super().__init__()
                del query_dim, num_frames, grid_hw, add_tokens, use_mask
                self.num_queries = int(num_queries)
                self.queries = nn.Parameter(torch.randn(1, self.num_queries, context_dim) * 0.02)
                self.q_cross_attn = CrossAttentionBlock(context_dim, num_heads)
                self.layers = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            d_model=context_dim,
                            nhead=num_heads,
                            dim_feedforward=int(context_dim * ffn_expansion_factor),
                            dropout=dropout,
                            batch_first=True,
                            norm_first=True,
                        )
                        for _ in range(num_layers)
                    ]
                )

            def forward(self, context: torch.Tensor) -> torch.Tensor:
                bsz, _, _, dim = context.shape
                queries = self.queries.expand(bsz, -1, -1).to(device=context.device, dtype=context.dtype)
                ctx = context.reshape(bsz, -1, dim)
                queries = self.q_cross_attn(queries, ctx)
                ctx = torch.cat([ctx, queries], dim=1)
                for layer in self.layers:
                    ctx = layer(ctx)
                return ctx[:, -self.num_queries :, :]

        modules_shim.QFormer_att = QFormer_att
        sys.modules[modules_name] = modules_shim

    pos_name = f"{package_name}.pos_embs"
    if pos_name not in sys.modules:
        _load_module(pos_name, utils_dir / "pos_embs.py")

    decoder_name = f"{package_name}.lam_decoder"
    decoder_module = sys.modules.get(decoder_name)
    if decoder_module is None:
        decoder_module = _load_module(decoder_name, utils_dir / "lam_decoder.py")
    encoder_name = f"{package_name}.lam_encoder"
    encoder_module = sys.modules.get(encoder_name)
    if encoder_module is None:
        encoder_module = _load_module(encoder_name, utils_dir / "lam_encoder.py")
    return decoder_module.LAMDecoder_v2, decoder_module.StatePredictor, encoder_module.LAMEncoder


LAMDecoder_v2, LaWAMStatePredictor, LAMEncoder = _load_lawam_components()


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentCurrentAdapter(nn.Module):
    """Maps pooled Qwen current context into the frozen V-JEPA latent space."""

    def __init__(self, qwen_dim: int, vj_dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(qwen_dim, vj_dim))
        self.adapter = MLPBlock(qwen_dim, hidden_dim, vj_dim, dropout=dropout)

    def forward(self, current_context: torch.Tensor) -> torch.Tensor:
        return self.adapter(current_context)


class StudentPredictor(nn.Module):
    """Predicts the deployable latent action from current Qwen context and state_0."""

    def __init__(self, qwen_dim: int, state_dim: int, latent_dim: int = 32, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or qwen_dim)
        self.predictor = MLPBlock(qwen_dim + state_dim, hidden_dim, latent_dim)

    def forward(self, current_context: torch.Tensor, state_0: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([current_context, state_0], dim=-1))


class TeacherEncoder(nn.Module):
    """Encodes privileged endpoint state with LaWAM's LAMEncoder/QFormer path."""

    def __init__(
        self,
        vj_dim: int,
        state_dim: int,
        latent_dim: int = 32,
        hidden_dim: int | None = None,
        num_layers: int = 4,
        num_heads: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        context_dim = int(hidden_dim or vj_dim)
        self.state_dim = int(state_dim)
        self.encoder = LAMEncoder(
            context_dim=context_dim,
            input_dim=int(vj_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
            num_frames=1,
            num_queries=1,
            grid_hw=(1, 1),
            add_state=True,
            max_state_dim=2 * self.state_dim,
            code_dim=int(latent_dim),
        )

    def forward(
        self,
        u_teacher: torch.Tensor,
        state_0: torch.Tensor,
        state_T: torch.Tensor,
        embodiment_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = int(u_teacher.shape[0])
        features = u_teacher.unsqueeze(1).unsqueeze(1)
        states = torch.cat([state_0, state_T], dim=-1).unsqueeze(1)
        if embodiment_id is None:
            embodiment_id = torch.zeros(bsz, device=u_teacher.device, dtype=torch.long)
        z = self.encoder(features=features, states=states, embodiment_id=embodiment_id)
        return z.squeeze(1)


class StateDeltaPredictor(nn.Module):
    """Optional LaWAM StatePredictor wrapper for L_state ablations."""

    def __init__(self, latent_dim: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.predictor = LaWAMStatePredictor(
            latent_dim=int(hidden_dim),
            max_state_dim=int(state_dim),
            code_dim=int(latent_dim),
        )

    def forward(
        self,
        z_pred: torch.Tensor,
        state_0: torch.Tensor,
        embodiment_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embodiment_id is None:
            embodiment_id = torch.zeros(z_pred.shape[0], device=z_pred.device, dtype=torch.long)
        return self.predictor(z_pred.unsqueeze(1), state_0, embodiment_id)


class SharedWorldDecoder(nn.Module):
    """
    Thin pooled-vector wrapper around LaWAM's LAMDecoder_v2.

    Public interface:
      u: [B, D_vj], z: [B, D_z] -> u_hat_T: [B, D_vj]
    """

    def __init__(
        self,
        vj_dim: int,
        latent_dim: int = 32,
        context_dim: int | None = None,
        num_layers: int = 6,
        num_heads: int = 16,
        dropout: float = 0.1,
        grid_hw: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        context_dim = int(context_dim or vj_dim)
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.decoder = LAMDecoder_v2(
            context_dim=context_dim,
            input_dim=int(vj_dim),
            num_queries=1,
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
            grid_hw=self.grid_hw,
            train_in_latent=True,
            code_dim=int(latent_dim),
            last_ln=True,
        )

    def forward(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        num_tokens = self.grid_hw[0] * self.grid_hw[1]
        features = u.unsqueeze(1).expand(-1, num_tokens, -1).unsqueeze(1)
        decoded = self.decoder(features, z.unsqueeze(1))
        if decoded.dim() == 4:
            decoded = decoded.squeeze(1)
        return decoded.mean(dim=1)
