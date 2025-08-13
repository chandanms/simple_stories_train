import inspect
import math
from typing import Any
from typing import cast as _cast
from typing import cast as t_cast

import torch
import torch.nn as nn
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import functional as F
from transformers import GPT2Config as HFGPT2Config
from transformers import GPT2LMHeadModel

from simple_stories_train.utils import print0

# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false


class GPT2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    flash_attention: bool = True


class NewGELU(nn.Module):
    def forward(self, input: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0)))
            )
        )


class CausalSelfAttention(nn.Module):
    bias: Tensor

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash_attention = config.flash_attention
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = True  # type: ignore
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(
        self,
        x: Float[Tensor, "batch pos d_model"],
    ) -> Float[Tensor, "batch pos d_model"]:
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash_attention:
            # use PyTorch SDPA
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = True  # type: ignore

    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Float[Tensor, "batch pos d_model"],
    ) -> Float[Tensor, "batch pos d_model"]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.wte: nn.Embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe: nn.Embedding = nn.Embedding(config.block_size, config.n_embd)
        _blocks: list[Block] = [Block(config) for _ in range(config.n_layer)]
        self.h_torch: nn.ModuleList = nn.ModuleList(_blocks)
        self.h: list[Block] = _blocks
        self.ln_f: nn.LayerNorm = nn.LayerNorm(config.n_embd)
        self.transformer: nn.ModuleDict = nn.ModuleDict(
            {
                "wte": self.wte,
                "wpe": self.wpe,
                "h": self.h_torch,
                "ln_f": self.ln_f,
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = True  # type: ignore
        self.wte.weight = self.lm_head.weight  # type: ignore[reportAttributeAccessIssue]

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = (
                0.02
                if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                else 0.02 / math.sqrt(2 * self.config.n_layer)
            )
            if not hasattr(module, "LLMC_SKIP_INIT"):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)  # type: ignore[arg-type]
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(
        self,
        idx: Float[Tensor, "batch pos"],
        targets: Float[Tensor, "batch pos vocab"] | None = None,
        return_logits: bool = True,
    ) -> tuple[
        Float[Tensor, "batch pos vocab"] | None,
        Float[Tensor, ""] | None,
    ]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)  # (b, t, n_embd)
        pos_emb = self.wpe(pos)  # (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        logits: Tensor = self.lm_head(x)
        loss: Tensor | None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            loss = None

        out_logits: Tensor | None = logits
        if not return_logits:
            out_logits = None

        return out_logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT2":
        """Loads pretrained GPT-2 model weights from Hugging Face."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel  # type: ignore

        print0(f"loading weights from pretrained gpt: {model_type}")
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config = GPT2Config(**_cast(dict[str, Any], config_args))
        model = GPT2(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd if not k.endswith(".attn.bias")]  # discard this mask / buffer

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [
            k for k in sd_hf if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))
        ]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
        zero_stage: int,
    ) -> torch.optim.Optimizer:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        print0(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optim_group = optim_groups[0]
            optimizer: torch.optim.Optimizer = ZeroRedundancyOptimizer(  # type: ignore[assignment]
                **optim_group,  # type: ignore[arg-type]
                optimizer_class=torch.optim.AdamW,
                lr=learning_rate,
                betas=betas,
                fused=use_fused,
            )
            optimizer.add_param_group(optim_groups[1])  # type: ignore[arg-type]
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=use_fused
            )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: Float[Tensor, "... pos"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> Float[Tensor, "... pos"]:
        # Keep track of whether input was 1D and ensure input has batch dimension
        is_1d = idx.dim() == 1
        if is_1d:
            idx = idx.unsqueeze(0)

        batch_size = idx.size(0)
        not_completed = torch.ones(batch_size, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            if not not_completed.any():
                break

            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            assert logits is not None
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
            else:
                probs = torch.zeros_like(logits)
                probs.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
            idx_next = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                not_completed = not_completed & (idx_next[:, -1] != eos_token_id)
                update_mask = not_completed.unsqueeze(-1)
                idx_next = torch.where(
                    update_mask, idx_next, torch.full_like(idx_next, eos_token_id)
                )

            idx = torch.cat((idx, idx_next), dim=1)

        if is_1d:
            idx = idx.squeeze(0)

        return idx


def convert_hf_gpt2_to_gpt2(hf_model: GPT2LMHeadModel) -> "GPT2":
    """Convert a HuggingFace GPT2LMHeadModel to our custom GPT2.

    Args:
        hf_model: HuggingFace GPT2LMHeadModel instance

    Returns:
        Our custom GPT2 model with weights copied from the HF model
    """
    hf_config = hf_model.config
    config = GPT2Config(
        block_size=hf_config.n_ctx,
        vocab_size=hf_config.vocab_size,
        n_layer=hf_config.n_layer,
        n_head=hf_config.n_head,
        n_embd=hf_config.n_embd,
        flash_attention=True,
    )
    model = GPT2(config)

    # Embeddings
    with torch.no_grad():
        model.wte.weight.copy_(hf_model.transformer.wte.weight)
        model.wpe.weight.copy_(hf_model.transformer.wpe.weight)

    # Blocks
    for i in range(config.n_layer):
        custom_block = model.h[i]
        hf_block = hf_model.transformer.h[i]

        # Layer norms
        with torch.no_grad():
            custom_block.ln_1.weight.copy_(t_cast(Tensor, hf_block.ln_1.weight))
            custom_block.ln_1.bias.copy_(t_cast(Tensor, hf_block.ln_1.bias))
            custom_block.ln_2.weight.copy_(t_cast(Tensor, hf_block.ln_2.weight))
            custom_block.ln_2.bias.copy_(t_cast(Tensor, hf_block.ln_2.bias))

        # Attention (transpose HF Conv1D weights to Linear)
        with torch.no_grad():
            custom_block.attn.c_attn.weight.copy_(t_cast(Tensor, hf_block.attn.c_attn.weight).T)
            custom_block.attn.c_attn.bias.copy_(t_cast(Tensor, hf_block.attn.c_attn.bias))

        with torch.no_grad():
            custom_block.attn.c_proj.weight.copy_(t_cast(Tensor, hf_block.attn.c_proj.weight).T)
            custom_block.attn.c_proj.bias.copy_(t_cast(Tensor, hf_block.attn.c_proj.bias))

        # MLP (transpose HF Conv1D weights to Linear)
        with torch.no_grad():
            custom_block.mlp.c_fc.weight.copy_(t_cast(Tensor, hf_block.mlp.c_fc.weight).T)
            custom_block.mlp.c_fc.bias.copy_(t_cast(Tensor, hf_block.mlp.c_fc.bias))

        with torch.no_grad():
            custom_block.mlp.c_proj.weight.copy_(t_cast(Tensor, hf_block.mlp.c_proj.weight).T)
            custom_block.mlp.c_proj.bias.copy_(t_cast(Tensor, hf_block.mlp.c_proj.bias))

    # Final ln_f
    with torch.no_grad():
        model.ln_f.weight.copy_(hf_model.transformer.ln_f.weight)
        model.ln_f.bias.copy_(hf_model.transformer.ln_f.bias)

    # LM head
    with torch.no_grad():
        model.lm_head.weight.copy_(hf_model.lm_head.weight)

    return model


def convert_gpt2_to_hf_gpt2(custom_model: GPT2) -> GPT2LMHeadModel:
    """Convert custom GPT-2 model to HuggingFace GPT2LMHeadModel.

    Args:
        custom_model: The custom GPT-2 model to convert

    Returns:
        The converted HuggingFace GPT2LMHeadModel
    """
    model_config: GPT2Config = custom_model.config

    hf_config = HFGPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.block_size,
        n_ctx=model_config.block_size,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        n_embd=model_config.n_embd,
        activation_function="gelu_new",
        n_inner=None,
        layer_norm_epsilon=1e-5,
        # Tie embeddings and lm_head as our implementation does
        tie_word_embeddings=True,
    )

    hf_model = GPT2LMHeadModel(hf_config)

    # Embeddings
    hf_model.transformer.wte.weight.data = custom_model.wte.weight.data
    hf_model.transformer.wpe.weight.data = custom_model.wpe.weight.data

    # Transformer blocks
    for i in range(model_config.n_layer):
        custom_block = custom_model.h[i]
        hf_block = hf_model.transformer.h[i]

        # LayerNorms
        hf_block.ln_1.weight.data = custom_block.ln_1.weight.data
        hf_block.ln_1.bias.data = custom_block.ln_1.bias.data
        hf_block.ln_2.weight.data = custom_block.ln_2.weight.data
        hf_block.ln_2.bias.data = custom_block.ln_2.bias.data

        # Attention projections: HF uses Conv1D (weight shape [in, out])
        # Our Linear weights are [out, in], so transpose when copying to HF
        hf_block.attn.c_attn.weight.data = custom_block.attn.c_attn.weight.data.t().contiguous()
        hf_block.attn.c_attn.bias.data = custom_block.attn.c_attn.bias.data

        hf_block.attn.c_proj.weight.data = custom_block.attn.c_proj.weight.data.t().contiguous()
        hf_block.attn.c_proj.bias.data = custom_block.attn.c_proj.bias.data

        # MLP projections
        hf_block.mlp.c_fc.weight.data = custom_block.mlp.c_fc.weight.data.t().contiguous()
        hf_block.mlp.c_fc.bias.data = custom_block.mlp.c_fc.bias.data

        hf_block.mlp.c_proj.weight.data = custom_block.mlp.c_proj.weight.data.t().contiguous()
        hf_block.mlp.c_proj.bias.data = custom_block.mlp.c_proj.bias.data

    # Final LayerNorm
    hf_model.transformer.ln_f.weight.data = custom_model.ln_f.weight.data
    hf_model.transformer.ln_f.bias.data = custom_model.ln_f.bias.data

    # LM head
    hf_model.lm_head.weight.data = custom_model.lm_head.weight.data

    hf_model.eval()
    return hf_model
