"""
train_mtp_c3.py -- LoRA fine-tune the C3 decoder with a shared-weight MTP head.

- Loads precomputed latents (from precompute_latents.py)
- Freezes the C3 decoder, applies LoRA to q_proj/v_proj
- Adds a single MTP head (2-layer MLP), unrolled K times per position
- Loss = standard next-token CE + (1/K) * mean(MTP offset CEs)
- Bypasses C3 encoder path by calling Qwen2Model.forward() directly

Auto-resume: on startup, automatically finds and resumes from the latest checkpoint
in --save_dir (highest step_XXXXX directory). No --resume_from flag needed.

Epoch-seeded sampler: LengthBucketSampler uses seed + epoch, so batch order is
reproducible across crashes — same epoch always produces the same batch order.
On resume, fast-forwards through already-processed batches in the current epoch
(data is in RAM, so this loop is essentially free — no GPU work).
"""

import argparse
import glob
import math
import os
import shutil
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
        return self.fc2(self.act(self.fc1(self.norm(x))))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LatentDataset(Dataset):
    """Loads all .pt shards into RAM. Each item: {latents, target_ids}.

    Accepts both old-style shard_*.pt and new-style rank*_shard_*.pt filenames.
    """

    def __init__(self, latent_dir):
        shard_paths = sorted(
            glob.glob(os.path.join(latent_dir, "shard_*.pt")) +
            glob.glob(os.path.join(latent_dir, "rank*_shard_*.pt"))
        )
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
    """Groups examples by target_ids length to minimize padding waste.

    Deterministic per epoch: shuffle seed = base_seed + epoch, so the same
    epoch always produces the same batch order regardless of crash/restart.
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        lengths = [len(d["target_ids"]) for d in dataset.data]
        self.sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])
        # Pre-build sorted batches (order within each batch is fixed)
        self._batches = []
        for i in range(0, len(self.sorted_indices), batch_size):
            b = self.sorted_indices[i:i + batch_size]
            if len(b) == batch_size:  # drop_last equivalent
                self._batches.append(b)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            perm = torch.randperm(len(self._batches), generator=g).tolist()
            batches = [self._batches[i] for i in perm]
        else:
            batches = self._batches
        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self._batches) * self.batch_size


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

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

def train_step(batch, model, mtp_head, prompt_ids, K, device, mtp_weight=1.0):
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

    total_loss = recon_loss + mtp_weight * sum(mtp_losses) / max(K, 1)
    return total_loss, recon_loss, mtp_losses


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(val_dataloader, model, mtp_head, prompt_ids, K, device,
                   max_batches=50):
    """Run forward passes on held-out data and return average metrics."""
    model.eval()
    mtp_head.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_mtp = [0.0] * K
    n_batches = 0

    for batch_idx, batch in enumerate(val_dataloader):
        if batch_idx >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, recon, mtp_losses = train_step(
                batch, model, mtp_head, prompt_ids, K, device)
        total_loss += loss.item()
        total_recon += recon.item()
        for k in range(K):
            total_mtp[k] += mtp_losses[k].item()
        n_batches += 1

    model.train()
    mtp_head.train()

    n = max(n_batches, 1)
    return total_loss / n, total_recon / n, [t / n for t in total_mtp]


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def find_latest_checkpoint(save_dir):
    """Scan save_dir for step_XXXXX directories, return path with highest step.

    Validates each candidate by loading training_state.pt — skips corrupted ones.
    Returns None if save_dir doesn't exist or contains no valid checkpoints.
    """
    if not os.path.exists(save_dir):
        return None
    candidates = []
    for name in os.listdir(save_dir):
        if name.startswith("step_") and name[5:].isdigit():
            step_num = int(name[5:])
            ckpt_path = os.path.join(save_dir, name)
            state_path = os.path.join(ckpt_path, "training_state.pt")
            if not os.path.exists(state_path):
                continue
            try:
                torch.load(state_path, map_location="cpu", weights_only=True)
                candidates.append((step_num, ckpt_path))
            except Exception:
                print(f"  Warning: corrupted checkpoint {ckpt_path}, skipping")
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def save_checkpoint(model, mtp_head, optimizer, scheduler,
                    step, epoch, batch_idx_in_epoch, args, save_dir):
    """Save LoRA adapter, MTP head, optimizer, and training state.

    Atomic: writes to a .tmp directory first, then renames. If the process
    crashes mid-save, the .tmp dir is ignored by find_latest_checkpoint.
    """
    tmp_dir = save_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    model.save_pretrained(os.path.join(tmp_dir, "lora_adapter"))
    torch.save(mtp_head.state_dict(), os.path.join(tmp_dir, "mtp_head.pt"))
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
        "batch_idx_in_epoch": batch_idx_in_epoch,
        "batch_size": args.batch_size,
        "mtp_k": args.mtp_k,
        "lr": args.lr,
    }, os.path.join(tmp_dir, "training_state.pt"))

    # Atomic swap — if save_dir exists (e.g. final ckpt matching periodic),
    # remove it first so rename succeeds.
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.rename(tmp_dir, save_dir)
    print(f"  Checkpoint saved to {save_dir}")


def load_checkpoint(model, mtp_head, optimizer, scheduler, checkpoint_dir, device):
    """Load a saved checkpoint. Returns (step, epoch, batch_idx_in_epoch).

    If loading fails (corrupted files), prints a warning and returns zeros
    so training can restart from scratch instead of crashing.
    """
    try:
        lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
        if os.path.exists(lora_dir):
            model.load_adapter(lora_dir, adapter_name="default")

        mtp_path = os.path.join(checkpoint_dir, "mtp_head.pt")
        if os.path.exists(mtp_path):
            mtp_head.load_state_dict(
                torch.load(mtp_path, map_location=device, weights_only=True))

        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        state = torch.load(state_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        return (state["step"], state.get("epoch", 0),
                state.get("batch_idx_in_epoch", 0))
    except Exception as e:
        print(f"  WARNING: failed to load checkpoint {checkpoint_dir}: {e}")
        print(f"  Starting from scratch.")
        return 0, 0, 0


# ---------------------------------------------------------------------------
# LR schedule: Warmup-Stable-Decay (WSD)
# ---------------------------------------------------------------------------

def get_wsd_schedule(optimizer, warmup_steps, total_steps,
                     stable_frac=0.8, min_lr_ratio=0.1):
    """Warmup-Stable-Decay schedule (Llama 3 / DeepSeek style).

    Phase 1 — Linear warmup:  0 → peak_lr over warmup_steps
    Phase 2 — Stable plateau: hold peak_lr for stable_frac of remaining steps
    Phase 3 — Cosine decay:   peak_lr → min_lr over the rest

    min_lr_ratio: final LR as fraction of peak (0.1 = decay to 10% of peak).
    """
    remaining = total_steps - warmup_steps
    stable_steps = int(remaining * stable_frac)
    decay_start = warmup_steps + stable_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < decay_start:
            return 1.0
        decay_steps = total_steps - decay_start
        progress = min(1.0, (step - decay_start) / max(1, decay_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MTP on C3 latent space")
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="liufanfanlff/C3-Context-Cascade-Compression")
    parser.add_argument("--mtp_k", type=int, default=3, choices=[1, 2, 3, 5, 10])
    parser.add_argument("--mtp_weight", type=float, default=0.3,
                        help="Weight for MTP loss relative to recon loss")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Final LR as fraction of peak (0.1 = decay to 10%%)")
    parser.add_argument("--stable_frac", type=float, default=0.8,
                        help="Fraction of post-warmup steps at peak LR before decay")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=50_000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints_real")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="Repeat the text: ")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for sampler reproducibility")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--val_latent_dir", type=str, default=None,
                        help="Latent dir for held-out set (skip validation if omitted)")
    parser.add_argument("--val_every", type=int, default=2000,
                        help="Run validation every N steps")
    parser.add_argument("--val_batches", type=int, default=50,
                        help="Max batches per validation run")
    args = parser.parse_args()

    device = torch.device("cuda")

    # --- Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True)

    prompt_ids = torch.tensor(
        tokenizer.encode(args.prompt), dtype=torch.long, device=device)
    print(f"Prompt: '{args.prompt}' -> {len(prompt_ids)} tokens")

    # --- Load C3 decoder ---
    print("Loading C3 model (decoder only)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
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

    # --- Dataset + sampler ---
    dataset = LatentDataset(args.latent_dir)
    sampler = LengthBucketSampler(
        dataset, args.batch_size, shuffle=True, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(sampler._batches)  # exact: pre-built batches count

    # --- Validation dataloader (optional) ---
    val_dataloader = None
    if args.val_latent_dir:
        val_dataset = LatentDataset(args.val_latent_dir)
        val_sampler = LengthBucketSampler(
            val_dataset, args.batch_size, shuffle=False, seed=0)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Validation: {len(val_dataset)} examples, "
              f"every {args.val_every} steps, {args.val_batches} batches/run")

    # --- Optimizer & scheduler ---
    trainable_params = (
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(mtp_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01,
                                   betas=(0.9, 0.95))
    scheduler = get_wsd_schedule(optimizer, args.warmup_steps, args.max_steps,
                                 stable_frac=args.stable_frac,
                                 min_lr_ratio=args.min_lr_ratio)
    decay_start = args.warmup_steps + int(
        (args.max_steps - args.warmup_steps) * args.stable_frac)
    print(f"LR schedule (WSD): warmup 0→{args.lr:.0e} [{args.warmup_steps} steps] | "
          f"stable [{args.warmup_steps}→{decay_start}] | "
          f"decay → {args.lr * args.min_lr_ratio:.0e} [{decay_start}→{args.max_steps}]")

    # --- Auto-resume: find latest checkpoint ---
    start_step = 0
    start_epoch = 0
    start_batch_idx = 0
    latest_ckpt = find_latest_checkpoint(args.save_dir)
    if latest_ckpt:
        print(f"Auto-resuming from {latest_ckpt}...")
        start_step, start_epoch, start_batch_idx = load_checkpoint(
            model, mtp_head, optimizer, scheduler, latest_ckpt, device)
        print(f"Resumed at step={start_step}, epoch={start_epoch}, "
              f"batch_idx={start_batch_idx}")

    # --- Training loop ---
    grad_accum = args.grad_accum_steps
    eff_batch = args.batch_size * grad_accum
    print(f"\nStarting training: K={args.mtp_k}, lr={args.lr}, "
          f"micro_batch={args.batch_size}, grad_accum={grad_accum}, "
          f"eff_batch={eff_batch}, max_steps={args.max_steps}")
    print(f"steps_per_epoch={steps_per_epoch // grad_accum}")
    print(f"{'='*80}")

    model.train()
    mtp_head.train()
    step = start_step
    running_loss = 0.0
    accum_loss = 0.0
    micro_count = 0
    t_start = time.time()

    epoch = start_epoch
    while step < args.max_steps:
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        micro_count = 0
        accum_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # Fast-forward: skip batches already processed in this epoch.
            # Data is in RAM, so enumerate is cheap — no GPU work happens here.
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            if step >= args.max_steps:
                break

            with torch.autocast("cuda", dtype=torch.bfloat16):
                total_loss, recon_loss, mtp_losses = train_step(
                    batch, model, mtp_head, prompt_ids, args.mtp_k, device,
                    mtp_weight=args.mtp_weight)

            (total_loss / grad_accum).backward()
            accum_loss += total_loss.item()
            micro_count += 1

            if micro_count < grad_accum:
                continue

            # --- Full optimizer step ---
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            running_loss += accum_loss / grad_accum
            accum_loss = 0.0
            micro_count = 0

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
                save_checkpoint(model, mtp_head, optimizer, scheduler,
                                step, epoch, batch_idx + 1, args, ckpt_dir)

            # --- Validation ---
            if val_dataloader and step % args.val_every == 0:
                t_val = time.time()
                val_loss, val_recon, val_mtp = run_validation(
                    val_dataloader, model, mtp_head, prompt_ids,
                    args.mtp_k, device, max_batches=args.val_batches)
                val_mtp_strs = ", ".join(
                    f"mtp_{k+1}={val_mtp[k]:.4f}" for k in range(args.mtp_k))
                print(f"  [VAL step={step}] loss={val_loss:.4f} | "
                      f"recon={val_recon:.4f} | {val_mtp_strs} | "
                      f"took {time.time()-t_val:.0f}s")

        epoch += 1

    # Final checkpoint (skip if this step was already saved by periodic checkpointing)
    ckpt_dir = os.path.join(args.save_dir, f"step_{step}")
    if not os.path.exists(ckpt_dir):
        save_checkpoint(model, mtp_head, optimizer, scheduler,
                        step, epoch, 0, args, ckpt_dir)
    print(f"\nTraining complete. Final step: {step}")


if __name__ == "__main__":
    main()
