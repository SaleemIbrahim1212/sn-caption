"""
Greedy decoding with per-step decoder attention weights.

Mirrors DecoderRNN / DualModalDecoderRNN math from model.py without changing src/.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# Match dataset.PAD_TOKEN / SOS_TOKEN / EOS_TOKEN (avoid importing dataset → torchtext at import time).
SOS_TOKEN = 1
EOS_TOKEN = 2


def _attn_weights(
    query: torch.Tensor, encoder_out: torch.Tensor, scale_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """query (B,1,H), encoder_out (B,T,H) -> weights (B,T), context (B,H)."""
    context = query @ encoder_out.permute(0, 2, 1) / (scale_dim**0.5)
    w = context.softmax(dim=2)
    ctx = (w @ encoder_out).squeeze(1)
    return w.squeeze(1), ctx


def greedy_decode_decoder_rnn_with_attention(
    decoder,
    features: torch.Tensor,
    encoder_outputs: torch.Tensor,
    max_seq_length: int = 70,
    scale_dim: int = 512,
) -> Tuple[List[int], List[torch.Tensor]]:
    """
    features: (B, D) pooled init; encoder_outputs: (B, T, H).
    Returns token ids (excluding EOS) and list of attention weights (T,) per generated token.
    """
    device = features.device
    sampled_ids: List[int] = []
    attn_per_step: List[torch.Tensor] = []

    x = decoder.ft_extactor_2(decoder.activation(decoder.dropout(decoder.ft_extactor_1(features))))
    x = torch.stack([x] * decoder.num_layers, dim=0)
    states = (x, x)

    word = decoder.embed(torch.tensor([[SOS_TOKEN]], device=device, dtype=torch.long)).squeeze(1)

    for _ in range(max_seq_length):
        query = states[0][-1].unsqueeze(1)
        w, final_context = _attn_weights(query, encoder_outputs, scale_dim)
        attn_per_step.append(w.detach().float().cpu()[0].clone())
        inputs = torch.cat([word, final_context], dim=1)
        hiddens, states = decoder.lstm(inputs.unsqueeze(1), states)
        logit = decoder.fc(hiddens.squeeze(1))
        predicted = int(logit.argmax(1).item())
        sampled_ids.append(predicted)
        if predicted == EOS_TOKEN:
            break
        word = decoder.embed(torch.tensor([[predicted]], device=device, dtype=torch.long)).squeeze(1)

    return sampled_ids, attn_per_step


def greedy_decode_dual_modal_with_attention(
    decoder,
    feat_a: torch.Tensor,
    feat_v: torch.Tensor,
    enc_a: torch.Tensor,
    enc_v: torch.Tensor,
    max_seq_length: int = 70,
) -> Tuple[List[int], List[torch.Tensor], List[torch.Tensor]]:
    """Returns token ids, per-step audio weights (Ta,), video weights (Tv,)."""
    device = feat_a.device
    sampled_ids: List[int] = []
    attn_a_steps: List[torch.Tensor] = []
    attn_v_steps: List[torch.Tensor] = []

    states_a, states_v = decoder.initial_states(feat_a, feat_v)
    word = decoder.embed(torch.tensor([[SOS_TOKEN]], device=device, dtype=torch.long)).squeeze(1)
    d = int(decoder.context_dim)

    for _ in range(max_seq_length):
        qa = states_a[0][-1].unsqueeze(1)
        qv = states_v[0][-1].unsqueeze(1)
        wa, ctx_a = _attn_weights(qa, enc_a, d)
        wv, ctx_v = _attn_weights(qv, enc_v, d)
        attn_a_steps.append(wa.detach().float().cpu()[0].clone())
        attn_v_steps.append(wv.detach().float().cpu()[0].clone())

        in_a = torch.cat([word, ctx_a], dim=1)
        in_v = torch.cat([word, ctx_v], dim=1)
        ha, states_a = decoder.lstm_a(in_a.unsqueeze(1), states_a)
        hv, states_v = decoder.lstm_v(in_v.unsqueeze(1), states_v)
        logits = decoder.fc(torch.cat([ha.squeeze(1), hv.squeeze(1)], dim=1))
        predicted = int(logits.argmax(1).item())
        sampled_ids.append(predicted)
        if predicted == EOS_TOKEN:
            break
        word = decoder.embed(torch.tensor([[predicted]], device=device, dtype=torch.long)).squeeze(1)

    return sampled_ids, attn_a_steps, attn_v_steps


def run_explain_baseline(model, feats_v: torch.Tensor, device: torch.device, max_len: int):
    """Video2Caption: encoder then degenerate T=1 attention (matches teacher-forcing path)."""
    model.eval()
    with torch.no_grad():
        pooled = model.encoder(feats_v.unsqueeze(0).to(device))
        enc = pooled.unsqueeze(1)
        ids, attn = greedy_decode_decoder_rnn_with_attention(
            model.decoder, pooled, enc, max_seq_length=max_len, scale_dim=512
        )
    return ids, attn, int(enc.size(1))


def run_explain_transformer(
    model,
    feats_v: Optional[torch.Tensor],
    feats_a: Optional[torch.Tensor],
    device: torch.device,
    max_len: int,
) -> Tuple[List[int], Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]], Dict[str, Any]]:
    """
    Returns (token_ids, attention_data, meta).
    attention_data is either list of (T,) tensors (single decoder) or (audio_list, video_list) for dual.
    """
    model.eval()
    pool = model.encoder.pool
    meta: Dict[str, Any] = {"pool": pool, "dual_decoder": bool(getattr(model, "use_dual_decoder", False))}

    with torch.no_grad():
        if pool == "Transformer_Video":
            fv = feats_v.unsqueeze(0).to(device) if feats_v is not None else None
            features, encoder_out = model.encoder(video_feats=fv)
        elif pool == "Transformer_Audio":
            fa = feats_a.unsqueeze(0).to(device) if feats_a is not None else None
            features, encoder_out = model.encoder(audio_feats=fa)
        else:
            fv = feats_v.unsqueeze(0).to(device) if feats_v is not None else None
            fa = feats_a.unsqueeze(0).to(device) if feats_a is not None else None
            features, encoder_out = model.encoder(audio_feats=fa, video_feats=fv)

        meta["encoder_seq_len"] = int(encoder_out.size(1))
        meta["fused_feature_dim"] = int(features.size(-1))

        if getattr(model, "use_dual_decoder", False):
            half_d = features.size(-1) // 2
            feat_a, feat_v = features[:, :half_d], features[:, half_d:]
            tm = encoder_out.size(1) // 2
            enc_a, enc_v = encoder_out[:, :tm], encoder_out[:, tm:]
            meta["audio_seq_len"] = int(enc_a.size(1))
            meta["video_seq_len"] = int(enc_v.size(1))
            ids, aa, av = greedy_decode_dual_modal_with_attention(
                model.decoder, feat_a, feat_v, enc_a, enc_v, max_seq_length=max_len
            )
            return ids, (aa, av), meta

        ids, attn = greedy_decode_decoder_rnn_with_attention(
            model.decoder, features, encoder_out, max_seq_length=max_len, scale_dim=512
        )
        return ids, attn, meta
