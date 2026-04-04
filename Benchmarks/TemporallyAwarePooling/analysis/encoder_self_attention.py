"""
Extract transformer encoder self-attention weights for analysis only.

PyTorch's ``TransformerEncoderLayer`` calls ``MultiheadAttention`` with
``need_weights=False`` and may use a fused fast path that skips Python
``_sa_block``. This module:

1. Disables the MHA fast path for the duration of the capture (global setting).
2. Temporarily replaces each layer's ``_sa_block`` with a wrapper that calls
   ``self_attn(..., need_weights=True)`` once and stores averaged head weights.

No changes are made to ``src/transformer.py``; patching applies only to in-memory
module instances used in this script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class EncoderAttentionCapture:
    """Per forward pass: list of (layer_index, weights_B_T_T)."""

    layers: List[Tuple[int, torch.Tensor]] = field(default_factory=list)

    def clear(self) -> None:
        self.layers.clear()


def _make_patched_sa_block(capture: EncoderAttentionCapture, layer_index: int):
    """Return a function to be bound as ``layer._sa_block`` via ``MethodType``."""

    def _sa_block(
        self: nn.TransformerEncoderLayer,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn_out, attn_w = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
            average_attn_weights=True,
        )
        if attn_w is not None:
            capture.layers.append((layer_index, attn_w.detach().float().cpu().clone()))
        return self.dropout1(attn_out)

    return _sa_block


def _iter_transformer_encoder_layers(encoder: nn.TransformerEncoder) -> List[nn.TransformerEncoderLayer]:
    return list(encoder.layers)


def _apply_sa_block_patches(
    encoder: nn.TransformerEncoder, capture: EncoderAttentionCapture
) -> List[Tuple[nn.TransformerEncoderLayer, Callable]]:
    """Return list of (layer, original _sa_block) for restoration."""
    restored: List[Tuple[nn.TransformerEncoderLayer, Callable]] = []
    for i, layer in enumerate(_iter_transformer_encoder_layers(encoder)):
        restored.append((layer, layer._sa_block))
        layer._sa_block = MethodType(_make_patched_sa_block(capture, i), layer)
    return restored


def _restore_sa_block_patches(pairs: List[Tuple[nn.TransformerEncoderLayer, Callable]]) -> None:
    for layer, orig in pairs:
        layer._sa_block = orig


def _run_video_branch(pooling_layer: nn.Module, video_feats: torch.Tensor) -> torch.Tensor:
    """Replicate Transformer_Video.forward up to transformer input (proj + pos)."""
    batch_video, _, _ = video_feats.shape
    pos = pooling_layer.embedding_video(
        torch.arange(0, video_feats.shape[1], device=video_feats.device)
    )
    if pooling_layer.training:
        mask = (
            torch.rand(batch_video, 1, 1, device=video_feats.device) > pooling_layer.pos_dropout_rate
        ).float()
        pos = pos * mask
    x = pooling_layer.video_proj(video_feats) + pos
    return x


def _run_audio_branch(pooling_layer: nn.Module, audio_feats: torch.Tensor) -> torch.Tensor:
    batch_audio, _, _ = audio_feats.shape
    pos = pooling_layer.embedding_audio(
        torch.arange(0, audio_feats.shape[1], device=audio_feats.device)
    )
    if pooling_layer.training:
        mask = (
            torch.rand(batch_audio, 1, 1, device=audio_feats.device) > pooling_layer.pos_dropout_rate
        ).float()
        pos = pos * mask
    x = pooling_layer.audio_proj(audio_feats) + pos
    return x


def capture_transformer_video_encoder_attention(
    pooling_layer: nn.Module,
    video_feats: torch.Tensor,
) -> Tuple[EncoderAttentionCapture, torch.Tensor, torch.Tensor]:
    """
    Run ``Transformer_Video`` encoder with attention capture.

    Returns (capture, video_token, sequence_out) matching ``forward`` outputs.
    """
    capture = EncoderAttentionCapture()
    encoder = pooling_layer.video_transformer
    patches = _apply_sa_block_patches(encoder, capture)
    try:
        x = _run_video_branch(pooling_layer, video_feats)
        x = encoder(x)
        video_token = x.mean(dim=1)
        return capture, video_token, x
    finally:
        _restore_sa_block_patches(patches)


def capture_transformer_audio_encoder_attention(
    pooling_layer: nn.Module,
    audio_feats: torch.Tensor,
) -> Tuple[EncoderAttentionCapture, torch.Tensor, torch.Tensor]:
    capture = EncoderAttentionCapture()
    encoder = pooling_layer.audio_transformer
    patches = _apply_sa_block_patches(encoder, capture)
    try:
        x = _run_audio_branch(pooling_layer, audio_feats)
        x = encoder(x)
        audio_token = x.mean(dim=1)
        return capture, audio_token, x
    finally:
        _restore_sa_block_patches(patches)


def capture_multimodal_encoder_attention(
    pooling_layer: nn.Module,
    audio_feats: torch.Tensor,
    video_feats: torch.Tensor,
    streams: str = "both",
) -> Dict[str, Any]:
    """
    For ``Transformer`` (multimodal): capture audio and/or video encoder attentions.

    ``streams``: ``"video"``, ``"audio"``, or ``"both"``.
    Returns dict with keys ``video``, ``audio`` (each value is capture dict or None).
    """
    out: Dict[str, Any] = {"video": None, "audio": None}
    if streams in ("video", "both"):
        cap_v = EncoderAttentionCapture()
        enc_v = pooling_layer.video_transformer
        p_v = _apply_sa_block_patches(enc_v, cap_v)
        try:
            x_v = _run_video_branch_for_multimodal(pooling_layer, video_feats)
            x_v = enc_v(x_v)
            x_v = x_v + pooling_layer.modality_embed.weight[1].view(1, 1, -1)
            out["video"] = {"capture": cap_v, "sequence": x_v}
        finally:
            _restore_sa_block_patches(p_v)
    if streams in ("audio", "both"):
        cap_a = EncoderAttentionCapture()
        enc_a = pooling_layer.audio_transformer
        p_a = _apply_sa_block_patches(enc_a, cap_a)
        try:
            x_a = _run_audio_branch_for_multimodal(pooling_layer, audio_feats)
            x_a = enc_a(x_a)
            x_a = x_a + pooling_layer.modality_embed.weight[0].view(1, 1, -1)
            out["audio"] = {"capture": cap_a, "sequence": x_a}
        finally:
            _restore_sa_block_patches(p_a)
    return out


def _run_video_branch_for_multimodal(pooling_layer: nn.Module, video_feats: torch.Tensor) -> torch.Tensor:
    """Same as Transformer.forward video half before transformer."""
    batch_video, _, _ = video_feats.shape
    pos_v = pooling_layer.embedding_video(
        torch.arange(0, video_feats.shape[1], device=video_feats.device)
    )
    if pooling_layer.training:
        mask = (
            torch.rand(batch_video, 1, 1, device=video_feats.device) > pooling_layer.pos_dropout_rate
        ).float()
        pos_v = pos_v * mask
    return pooling_layer.video_proj(video_feats) + pos_v


def _run_audio_branch_for_multimodal(pooling_layer: nn.Module, audio_feats: torch.Tensor) -> torch.Tensor:
    batch_audio, _, _ = audio_feats.shape
    pos_a = pooling_layer.embedding_audio(
        torch.arange(0, audio_feats.shape[1], device=audio_feats.device)
    )
    if pooling_layer.training:
        mask = (
            torch.rand(batch_audio, 1, 1, device=audio_feats.device) > pooling_layer.pos_dropout_rate
        ).float()
        pos_a = pos_a * mask
    return pooling_layer.audio_proj(audio_feats) + pos_a


def capture_to_jsonable(capture: EncoderAttentionCapture) -> List[Dict[str, Any]]:
    """Batch dim 1: store [T,T] per layer as nested list."""
    rows = []
    for layer_idx, w in capture.layers:
        arr = w.squeeze(0).numpy().tolist()  # (T, T)
        rows.append({"layer": layer_idx, "attention": arr})
    return rows


class FastpathGuard:
    """Disable MHA fast path while active (restores previous flag on exit)."""

    def __init__(self):
        self._prev: Optional[bool] = None

    def __enter__(self):
        if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
            self._prev = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
        return self

    def __exit__(self, *args):
        if self._prev is not None:
            torch.backends.mha.set_fastpath_enabled(self._prev)
