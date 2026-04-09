# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

This function is generated semi-automatically.

Qwen3 MTP (Multi-Token Prediction) layer.

Mirrors vllm/model_executor/models/qwen3_next_mtp.py using transformers classes.

Reuse:
  Qwen3MoeRMSNorm, Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock: unchanged
  Qwen3MoeRotaryEmbedding: owned by MTPLayer with partial_rotary_factor=1.0
  Qwen3MTPAttention: thin subclass fixing o_proj size

Two non-obvious config values required:

1. partial_rotary_factor=1.0 (not 0.25 like the main model).
   apply_rotary_pos_emb does q*cos + rotate_half(q)*sin with no internal slicing,
   so cos/sin must be full head_dim=256. Factor=1.0 makes MTP rotary_emb produce
   256-dim embeddings and avoids the "size of tensor a (256) must match b (64)" error.

2. _attn_implementation="eager".
   Qwen3MoeAttention.forward does ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
   which raises KeyError: None if left unset.

"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)


class Qwen3MTPAttention(Qwen3MoeAttention):
    """
    Identical to Qwen3MoeAttention except o_proj has fan-in of 16*head_dim
    instead of 32*head_dim, matching the checkpoint weight [2048, 4096].
    """
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.o_proj = nn.Linear(
            (config.num_attention_heads // 2) * self.head_dim,
            config.hidden_size, bias=False
        )

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_value=None, cache_position=None, **kwargs):
        real_o_proj, self.o_proj = self.o_proj, nn.Identity()
        try:
            out, weights = super().forward(
                hidden_states, position_embeddings, attention_mask,
                past_key_value=past_key_value, cache_position=cache_position,
                **kwargs,
            )
        finally:
            self.o_proj = real_o_proj
        n = (self.config.num_attention_heads // 2) * self.head_dim
        return real_o_proj(out[..., :n]), weights


class Qwen3MTPLayer(nn.Module):
    """
    Qwen3 Multi-Token Prediction head.

    Forward (vLLM-style, shift=False, default):
        inputs_embeds = embed(input_ids)
        h = fc(cat([pre_fc_norm_embedding(inputs_embeds),
                    pre_fc_norm_hidden(hidden_states)], dim=-1))
        position_embeddings = rotary_emb(h, position_ids)
        h = decoder_layer(h, position_embeddings=position_embeddings)
        return norm(h)

    For offline calibration use shift=True to align hidden[t] with embed[t+1].
    """

    def __init__(self, config: Qwen3MoeConfig, state_dict: dict):
        super().__init__()
        self.config = config
        H, eps = config.hidden_size, config.rms_norm_eps

        self.pre_fc_norm_hidden    = Qwen3MoeRMSNorm(H, eps)
        self.pre_fc_norm_embedding = Qwen3MoeRMSNorm(H, eps)
        self.fc         = nn.Linear(H * 2, H, bias=False)
        self.norm       = Qwen3MoeRMSNorm(H, eps)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

        self.layer = Qwen3MoeDecoderLayer(config, layer_idx=0)
        self.layer.self_attn = Qwen3MTPAttention(config, layer_idx=0)
        self._load_weights(state_dict)

    def _load_weights(self, state_dict: dict):
        own = self.state_dict()
        loaded, missing, mismatched = [], [], []
        for name, param in own.items():
            ckpt_key = f"mtp.{name}".replace("mtp.layer.", "mtp.layers.0.")
            if ckpt_key in state_dict:
                src = state_dict[ckpt_key]
                if src.shape != param.shape:
                    mismatched.append((ckpt_key, tuple(src.shape), tuple(param.shape)))
                else:
                    own[name].copy_(src)
                    loaded.append(ckpt_key)
            else:
                missing.append(ckpt_key)
        self.load_state_dict(own)
        print(f"[MTP] loaded={len(loaded)}  missing={len(missing)}  "
              f"mismatched={len(mismatched)}")
        for k, cs, ms in mismatched:
            print(f"  MISMATCH {k}: ckpt={cs} model={ms}")

    def forward(
        self,
        hidden_states: torch.Tensor,                     # [B, L, H]
        input_ids: torch.Tensor = None,                  # [B, L]
        embedding_weight: torch.Tensor   = None,         # [vocab, H]
        inputs_embeds: torch.Tensor   = None,            # [B, L, H]
        position_ids: Optional[torch.Tensor]   = None,   # [B, L]; auto-generated if None
        attention_mask: Optional[torch.Tensor] = None,
        lm_head_weight: Optional[torch.Tensor] = None,
        shift: bool = False,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        if inputs_embeds is None:
            inputs_embeds = F.embedding(input_ids, embedding_weight)

        if shift:
            hidden_states = hidden_states[:, :-1]
            inputs_embeds = inputs_embeds[:, 1:]
            if attention_mask is not None and attention_mask.shape[-1] == L:
                attention_mask = attention_mask[:, :, :-1, :-1]
            L = L - 1

        if position_ids is None:
            position_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0)

        hidden_states = self.fc(torch.cat([
            self.pre_fc_norm_embedding(inputs_embeds),
            self.pre_fc_norm_hidden(hidden_states),
        ], dim=-1)).to(hidden_states)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

        hidden_states = self.norm(hidden_states)
        return F.linear(hidden_states, lm_head_weight) if lm_head_weight is not None \
               else hidden_states

    def reduce_experts(
        self,
        new_num_experts: int,
        keep_indices: Optional[List[int]] = None,
    ):
        """Prune routed experts (e.g. 512 → 384). keep_indices=None keeps first N."""
        if keep_indices is None:
            keep_indices = list(range(new_num_experts))
        assert len(keep_indices) == new_num_experts

        moe: Qwen3MoeSparseMoeBlock = self.layer.mlp
        old_n = len(moe.experts)
        assert max(keep_indices) < old_n

        new_gate = nn.Linear(
            self.config.hidden_size, new_num_experts, bias=False,
            device=moe.gate.weight.device, dtype=moe.gate.weight.dtype,
        )
        new_gate.weight = nn.Parameter(moe.gate.weight.data[keep_indices])
        moe.gate        = new_gate
        moe.experts     = nn.ModuleList([moe.experts[i] for i in keep_indices])
        moe.num_experts = new_num_experts
        self.config.num_experts = new_num_experts
        print(f"[MTP] experts: {old_n} → {new_num_experts}")

    def export_state_dict(self) -> dict:
        """State dict with checkpoint-format 'mtp.layers.0.*' keys."""
        return {
            f"mtp.{k}".replace("mtp.layer.", "mtp.layers.0."): v.clone()
            for k, v in self.state_dict().items()
        }


def build_mtp_layer(state_dict: dict, model='Qwen/Qwen3-Next-80B-A3B-Instruct', **config_overrides) -> Qwen3MTPLayer:
    """
    Build from a state_dict with 'mtp.*' keys.
    Defaults match Qwen3-Next-80B-A3B-Instruct/config.json.
    """
    fc_w = state_dict.get("mtp.fc.weight")
    if fc_w is None:
        raise KeyError("'mtp.fc.weight' not found — keys must have 'mtp.' prefix")

    if isinstance(model, str):
        cfg = Qwen3MoeConfig.from_pretrained(model).to_dict()
    else:
        # create a copy of model.config
        cfg = model.config.to_dict()

    cfg.update(config_overrides)
    cfg['partial_rotary_factor'] = 1.0
    cfg['_attn_implementation'] = "eager"
    cfg['num_attention_heads'] = 32
    # print(cfg)
    return Qwen3MTPLayer(Qwen3MoeConfig.from_dict(cfg), state_dict)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, L, H  = 2, 16, 2048
    vocab     = 151_936
    n_exp, top_k, int_sz = 8, 4, 512
    nh, nk, hd, o_h      = 32, 2, 256, 16

    sd = {
        "mtp.fc.weight":                    torch.randn(H, H*2) * 0.01,
        "mtp.norm.weight":                  torch.ones(H),
        "mtp.pre_fc_norm_hidden.weight":    torch.ones(H),
        "mtp.pre_fc_norm_embedding.weight": torch.ones(H),
        "mtp.layers.0.input_layernorm.weight":          torch.ones(H),
        "mtp.layers.0.post_attention_layernorm.weight": torch.ones(H),
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(nh*hd,   H)*0.01,
        "mtp.layers.0.self_attn.k_proj.weight": torch.randn(nk*hd,   H)*0.01,
        "mtp.layers.0.self_attn.v_proj.weight": torch.randn(nk*hd,   H)*0.01,
        "mtp.layers.0.self_attn.o_proj.weight": torch.randn(H, o_h*hd)*0.01,
        "mtp.layers.0.self_attn.q_norm.weight": torch.ones(hd),
        "mtp.layers.0.self_attn.k_norm.weight": torch.ones(hd),
        "mtp.layers.0.mlp.gate.weight":         torch.randn(n_exp, H)*0.01,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(int_sz, H)*0.01,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight":   torch.randn(int_sz, H)*0.01,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(H, int_sz)*0.01,
        "mtp.layers.0.mlp.shared_expert_gate.weight":      torch.randn(1, H)*0.01,
        **{f"mtp.layers.0.mlp.experts.{i}.gate_proj.weight": torch.randn(int_sz, H)*0.01
           for i in range(n_exp)},
        **{f"mtp.layers.0.mlp.experts.{i}.up_proj.weight":   torch.randn(int_sz, H)*0.01
           for i in range(n_exp)},
        **{f"mtp.layers.0.mlp.experts.{i}.down_proj.weight": torch.randn(H, int_sz)*0.01
           for i in range(n_exp)},
    }

    mtp = build_mtp_layer(sd, num_experts=n_exp, num_experts_per_tok=top_k)
    mtp.eval()

    assert isinstance(mtp.layer.input_layernorm, Qwen3MoeRMSNorm)
    assert isinstance(mtp.layer.mlp, Qwen3MoeSparseMoeBlock)
    assert isinstance(mtp.rotary_emb, Qwen3MoeRotaryEmbedding)
    print("✓ component types")

    hidden = torch.randn(B, L, H) * 0.1
    ids    = torch.randint(0, vocab, (B, L))
    emb_w  = torch.randn(vocab, H) * 0.01
    lm_w   = torch.randn(vocab, H) * 0.01

    with torch.no_grad():
        out = mtp(hidden, ids, emb_w, lm_head_weight=lm_w, shift=False)
        assert out.shape == (B, L, vocab), out.shape
        print(f"✓ shift=False: {out.shape}")

        out = mtp(hidden, ids, emb_w, lm_head_weight=lm_w, shift=True)
        assert out.shape == (B, L-1, vocab), out.shape
        print(f"✓ shift=True:  {out.shape}")

    mtp.reduce_experts(4, keep_indices=[0, 2, 4, 6])
    with torch.no_grad():
        out2 = mtp(hidden, ids, emb_w, lm_head_weight=lm_w, shift=True)
        assert out2.shape == (B, L-1, vocab)
    print(f"✓ reduce_experts(4): {out2.shape}")

    new_sd = mtp.export_state_dict()
    assert any("mtp.layers.0.self_attn" in k for k in new_sd)
    exp_ids = {int(k.split("experts.")[1].split(".")[0])
               for k in new_sd if "mlp.experts." in k}
    assert exp_ids == {0, 1, 2, 3}
    print("✓ export keys correct")
    print("\nAll tests passed!")
