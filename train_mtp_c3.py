"""
train_mtp_c3.py -- LoRA fine-tune the C3 decoder with a shared-weight MTP head.

- Loads precomputed latents (from precompute_latents.py)
- Freezes the C3 decoder, applies LoRA to q_proj/v_proj
- Adds a single MTP head (2-layer MLP), unrolled K times per position
- Loss = standard next-token CE + (1/K) * mean(MTP offset CEs)
- Bypasses C3 encoder path by calling Qwen2Model.forward() directly
"""

import argparse
import glob
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# MTP Head
# ---------------------------------------------------------------------------

class MTPHead(nn.Module):
    """Shared-weight multi-token prediction head.

    2-layer MLP with LayerNorm, applied autoregressively K times per position.
    At depth k:
        z_1 = head(h)
        z_k = head(z_{k-1} + h)   for k >= 2
    Each z_k is projected to vocab logits via the (frozen) LM head.
    """

    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, S, D] or [B, D]
        return self.fc2(self.act(self.fc1(self.norm(x))))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LatentDataset(Dataset):
    """Loads all .pt shards into RAM. Each item: {latents, target_ids}."""

    def __init__(self, latent_dir):
        shard_paths = sorted(glob.glob(os.path.join(latent_dir, "shard_*.pt")))
        assert shard_paths, f"No shards found in {latent_dir}"
        print(f"Loading {len(shard_paths)} shards from {latent_dir}...")
        self.data = []
        for path in shard_paths:
            shard = torch.load(path, map_location="cpu", weights_only=True)
            self.data.extend(shard)
        print(f"Loaded {len(self.data)} examples into RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LengthBucketSampler(Sampler):
    """Groups examples by target_ids length to minimize padding waste."""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Sort indices by target sequence length
        self.lengths = [len(d["target_ids"]) for d in dataset.data]
        self.sorted_indices = sorted(range(len(dataset)), key=lambda i: self.lengths[i])

    def __iter__(self):
        # Create batches from sorted indices
        batches = []
        for i in range(0, len(self.sorted_indices), self.batch_size):
            batch = self.sorted_indices[i:i + self.batch_size]
            batches.append(batch)
        if self.shuffle:
            # Shuffle the order of batches (not within batches)
            g = torch.Generator()
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.sorted_indices)


def collate_fn(batch):
    """Pad target_ids to max length in batch with -100, stack latents."""
    latents = torch.stack([b["latents"] for b in batch])  # [B, 32, 2048]
    target_ids_list = [b["target_ids"] for b in batch]
    max_len = max(t.shape[0] for t in target_ids_list)

    target_ids = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, t in enumerate(target_ids_list):
        target_ids[i, :t.shape[0]] = t

    return {"latents": latents, "target_ids": target_ids}


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(batch, model, mtp_head, prompt_ids, K, device):
    """Single training step. Returns total_loss, recon_loss, list of mtp_k losses."""
    latents = batch["latents"].to(device, dtype=torch.bfloat16)       # [B, 32, 2048]
    target_ids = batch["target_ids"].to(device)                       # [B, L]
    B = latents.shape[0]

    # Access internals through peft wrapper
    base_model = model.base_model.model          # C3QwenForCausalLM
    embed_fn = base_model.model.embed_tokens      # decoder embedding table
    lm_head = base_model.lm_head                  # frozen LM head
    decoder_backbone = base_model.model            # C3QwenModel (LoRA active)
    vocab_size = base_model.config.vocab_size

    # Embed prompt tokens
    P = prompt_ids.shape[0]
    prompt_embeds = embed_fn(prompt_ids).unsqueeze(0).expand(B, -1, -1)  # [B, P, 2048]

    # Embed target tokens (clamp -100 padding to 0 for embedding lookup)
    safe_ids = target_ids.clamp(min=0)
    target_embeds = embed_fn(safe_ids)  # [B, L, 2048]

    # Construct full input embeddings: [latents | prompt | target]
    inputs_embeds = torch.cat([latents, prompt_embeds, target_embeds], dim=1)

    # Attention mask: 1 for real positions, 0 for padding
    L = target_ids.shape[1]
    latent_mask = torch.ones(B, 32, device=device, dtype=torch.long)
    prompt_mask = torch.ones(B, P, device=device, dtype=torch.long)
    target_mask = (target_ids != -100).long()
    attention_mask = torch.cat([latent_mask, prompt_mask, target_mask], dim=1)

    # Labels: -100 for latent+prompt positions, actual IDs for targets
    ignore_prefix = torch.full((B, 32 + P), -100, device=device, dtype=torch.long)
    labels = torch.cat([ignore_prefix, target_ids], dim=1)  # [B, 32+P+L]

    # Forward through Qwen2 backbone (BYPASS C3 encoder logic)
    outputs = Qwen2Model.forward(
        decoder_backbone,
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    H = outputs.last_hidden_state  # [B, S, 2048] where S = 32 + P + L

    # --- Standard next-token CE (reconstruction loss) ---
    logits = lm_head(H)  # [B, S, V]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    recon_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # --- MTP losses ---
    mtp_losses = []
    Z = H  # will be overwritten
    for k in range(K):
        if k == 0:
            Z = mtp_head(H)
        else:
            Z = mtp_head(Z + H)

        mtp_logits = lm_head(Z)  # [B, S, V]
        shift = k + 2  # MTP depth k predicts token at position i + k + 2
        if shift >= labels.shape[1]:
            # Sequence too short for this depth, skip
            mtp_losses.append(torch.tensor(0.0, device=device))
            continue
        s_logits = mtp_logits[:, :-shift, :].contiguous()
        s_labels = labels[:, shift:].contiguous()
        mtp_k_loss = F.cross_entropy(
            s_logits.view(-1, vocab_size),
            s_labels.view(-1),
            ignore_index=-100,
        )
        mtp_losses.append(mtp_k_loss)

    total_loss = recon_loss + sum(mtp_losses) / max(K, 1)
    return total_loss, recon_loss, mtp_losses


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(model, mtp_head, optimizer, scheduler, step, args, save_dir):
    """Save LoRA adapter, MTP head, optimizer, and training state."""
    os.makedirs(save_dir, exist_ok=True)
    # LoRA adapter
    model.save_pretrained(os.path.join(save_dir, "lora_adapter"))
    # MTP head
    torch.save(mtp_head.state_dict(), os.path.join(save_dir, "mtp_head.pt"))
    # Training state
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "mtp_k": args.mtp_k,
        "lr": args.lr,
    }, os.path.join(save_dir, "training_state.pt"))
    print(f"  Checkpoint saved to {save_dir}")


def load_checkpoint(model, mtp_head, optimizer, scheduler, checkpoint_dir, device):
    """Load a saved checkpoint. Returns the step number."""
    from peft import PeftModel
    # LoRA adapter
    lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
    if os.path.exists(lora_dir):
        model.load_adapter(lora_dir, adapter_name="default")
    # MTP head
    mtp_path = os.path.join(checkpoint_dir, "mtp_head.pt")
    if os.path.exists(mtp_path):
        mtp_head.load_state_dict(torch.load(mtp_path, map_location=device, weights_only=True))
    # Training state
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        return state["step"]
    return 0


# ---------------------------------------------------------------------------
# Cosine schedule with warmup
# ---------------------------------------------------------------------------

def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MTP on C3 latent space")
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="liufanfanlff/C3-Context-Cascade-Compression")
    parser.add_argument("--mtp_k", type=int, default=5, choices=[1, 5, 10])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=50_000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="Repeat the text: ")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint directory to resume from")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda")

    # --- Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True)

    # Tokenize the fixed prompt
    prompt_ids = torch.tensor(
        tokenizer.encode(args.prompt), dtype=torch.long, device=device)
    print(f"Prompt: '{args.prompt}' -> {len(prompt_ids)} tokens")

    # --- Load C3 decoder ---
    print("Loading C3 model (decoder only)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Free encoder components (not needed for training)
    model.model.llm1 = None
    model.model.Q = None
    model.model.mm_projector = None
    torch.cuda.empty_cache()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- MTP head ---
    hidden_dim = model.base_model.model.config.hidden_size  # 2048
    mtp_head = MTPHead(hidden_dim=hidden_dim).to(device=device, dtype=torch.bfloat16)
    print(f"MTP head params: {sum(p.numel() for p in mtp_head.parameters()):,}")

    # --- Dataset ---
    dataset = LatentDataset(args.latent_dir)
    sampler = LengthBucketSampler(dataset, args.batch_size, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # --- Optimizer & scheduler ---
    trainable_params = (
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(mtp_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, args.max_steps)

    # --- Resume from checkpoint ---
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}...")
        start_step = load_checkpoint(
            model, mtp_head, optimizer, scheduler, args.resume_from, device)
        print(f"Resumed at step {start_step}")

    # --- Training loop ---
    print(f"\nStarting training: K={args.mtp_k}, lr={args.lr}, "
          f"batch_size={args.batch_size}, max_steps={args.max_steps}")
    print(f"{'='*80}")

    model.train()
    mtp_head.train()
    step = start_step
    epoch = 0
    running_loss = 0.0
    t_start = time.time()

    while step < args.max_steps:
        epoch += 1
        for batch in dataloader:
            if step >= args.max_steps:
                break

            optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                total_loss, recon_loss, mtp_losses = train_step(
                    batch, model, mtp_head, prompt_ids, args.mtp_k, device)

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            step += 1
            running_loss += total_loss.item()

            # --- Logging ---
            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                elapsed = time.time() - t_start
                lr_now = scheduler.get_last_lr()[0]
                mtp_strs = ", ".join(
                    f"mtp_{k+1}={mtp_losses[k].item():.4f}"
                    for k in range(len(mtp_losses))
                )
                print(
                    f"step={step:6d} | loss={avg_loss:.4f} | "
                    f"recon={recon_loss.item():.4f} | {mtp_strs} | "
                    f"lr={lr_now:.2e} | gnorm={grad_norm:.2f} | "
                    f"elapsed={elapsed:.0f}s"
                )
                running_loss = 0.0

            # --- Checkpointing ---
            if step % args.save_every == 0:
                ckpt_dir = os.path.join(args.save_dir, f"step_{step}")
                save_checkpoint(model, mtp_head, optimizer, scheduler, step,
                                args, ckpt_dir)

    # Final checkpoint
    ckpt_dir = os.path.join(args.save_dir, f"step_{step}")
    save_checkpoint(model, mtp_head, optimizer, scheduler, step, args, ckpt_dir)
    print(f"\nTraining complete. Final step: {step}")


if __name__ == "__main__":
    main()
