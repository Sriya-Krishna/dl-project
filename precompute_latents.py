"""
precompute_latents.py — Run the C3 encoder on all training/eval data and save
projected latent vectors as .pt shards.

Speedups:
  - Pre-tokenize ALL examples upfront (no tokenizer in the GPU hot loop)
  - Sort by context length for tight batching (minimal padding waste)
  - flash_attention_2 encoder
  - torch.compile on the encoder forward (--compile flag)
  - Async shard saving in a background thread (GPU never waits for disk)
  - Multi-GPU: --rank / --world_size splits work across N independent processes

Multi-GPU usage (4 GPUs, run each in a separate terminal or with &):
    CUDA_VISIBLE_DEVICES=0 python precompute_latents.py ... --rank 0 --world_size 4
    CUDA_VISIBLE_DEVICES=1 python precompute_latents.py ... --rank 1 --world_size 4
    CUDA_VISIBLE_DEVICES=2 python precompute_latents.py ... --rank 2 --world_size 4
    CUDA_VISIBLE_DEVICES=3 python precompute_latents.py ... --rank 3 --world_size 4
Each rank writes its own shard files (prefixed with rank_N_) to the same output_dir.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name, device="cuda", compile_encoder=False):
    """Load C3 model, swap encoder to flash_attention_2, optionally compile."""
    print(f"Loading C3 model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Swap encoder for flash_attention_2 version
    print("Reloading encoder with flash_attention_2...")
    encoder_fa = Qwen2ForCausalLM.from_pretrained(
        model_name,
        subfolder="llm1",
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        use_safetensors=True,
    )
    model.model.llm1 = encoder_fa
    del encoder_fa
    torch.cuda.empty_cache()

    model.eval()
    model.requires_grad_(False)

    if compile_encoder:
        print("Compiling encoder with torch.compile (first batch will be slow)...")
        model.model.llm1 = torch.compile(
            model.model.llm1, mode="reduce-overhead", fullgraph=False)

    print("Model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Pre-tokenize all data upfront (CPU, done once)
# ---------------------------------------------------------------------------

def pretokenize_all(tokenizer, examples, latent_len, rank, world_size,
                    chunk_size=2000):
    """
    Tokenize every example on CPU. This rank only processes its slice.

    Context IDs = raw text token IDs + [<img>, <imgpad>*32, </img>]
    Appending suffix IDs directly is ~4x faster than string-building.

    Returns:
        ctx_ids_list   : list of list[int]
        target_ids_list: list of torch.LongTensor
        local_indices  : original indices this rank owns (for shard naming)
    """
    im_start = tokenizer.convert_tokens_to_ids("<img>")
    im_patch = tokenizer.convert_tokens_to_ids("<imgpad>")
    im_end   = tokenizer.convert_tokens_to_ids("</img>")
    suffix   = [im_start] + [im_patch] * latent_len + [im_end]

    # Slice belonging to this rank
    local_indices = list(range(rank, len(examples), world_size))
    local_examples = [examples[i] for i in local_indices]
    texts = [ex["text"] for ex in local_examples]

    ctx_ids_list    = []
    target_ids_list = []
    t0 = time.time()

    for start in range(0, len(texts), chunk_size):
        chunk = texts[start: start + chunk_size]
        enc = tokenizer(chunk, truncation=False, add_special_tokens=False)
        for ids in enc["input_ids"]:
            target_ids_list.append(torch.tensor(ids, dtype=torch.long))
            ctx_ids_list.append(ids + suffix)

        done = start + len(chunk)
        if done % 50_000 == 0 or done == len(texts):
            print(f"  [rank {rank}] Pre-tokenized {done}/{len(texts)}"
                  f"  ({time.time()-t0:.0f}s)")

    return ctx_ids_list, target_ids_list, local_indices


# ---------------------------------------------------------------------------
# Encoder batch forward
# ---------------------------------------------------------------------------

def encode_batch(model, ctx_ids_batch, device="cuda"):
    """
    Forward one GPU batch. Returns latents_proj [B, 32, 2048] on CPU.
    """
    config     = model.config
    latent_len = config.latent_token_len  # 32
    im_start   = config.im_start_token   # 151857
    encoder    = model.model.llm1
    Q_weight   = model.model.Q.weight    # [32, 1536]
    mm_proj    = model.model.mm_projector

    B       = len(ctx_ids_batch)
    lengths = [len(ids) for ids in ctx_ids_batch]
    max_len = max(lengths)

    # Build padded tensors
    input_ids     = torch.zeros((B, max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, (ids, L) in enumerate(zip(ctx_ids_batch, lengths)):
        t = torch.tensor(ids, dtype=torch.long)
        input_ids[i, :L]     = t
        attention_mask[i, :L] = 1

    # <img> position is deterministic: lengths[i] - latent_len - 2
    # (suffix = <img> + 32*<imgpad> + </img>, so <img> is 34 from the end)
    img_positions = torch.tensor(
        [L - latent_len - 2 for L in lengths], dtype=torch.long, device=device)

    # Embed tokens, then scatter Q.weight into query positions (vectorized)
    context_embeds = encoder.model.embed_tokens(input_ids)  # [B, S, 1536]
    Q_cast = Q_weight.to(device=device, dtype=context_embeds.dtype)  # [32, 1536]
    # Build a [B, 32] index tensor of positions to overwrite
    offsets = torch.arange(1, latent_len + 1, device=device)          # [32]
    query_pos = img_positions.unsqueeze(1) + offsets.unsqueeze(0)      # [B, 32]
    # Expand Q_cast to [B, 32, 1536] and scatter in one shot
    context_embeds[
        torch.arange(B, device=device).unsqueeze(1).expand(B, latent_len),
        query_pos,
    ] = Q_cast.unsqueeze(0).expand(B, -1, -1)

    # Encoder forward
    with torch.inference_mode():
        out = encoder(
            input_ids=None,
            inputs_embeds=context_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    last_hidden = out["hidden_states"][-1]  # [B, S, 1536]

    # Extract query positions (vectorized)
    latents_raw = last_hidden[
        torch.arange(B, device=device).unsqueeze(1).expand(B, latent_len),
        query_pos,
    ]  # [B, 32, 1536]

    # Cast + project
    latents_raw = latents_raw.to(dtype=mm_proj.weight.dtype)
    with torch.inference_mode():
        latents_proj = mm_proj(latents_raw)  # [B, 32, 2048]

    return latents_proj.cpu()


# ---------------------------------------------------------------------------
# Async shard saving
# ---------------------------------------------------------------------------

_save_executor = ThreadPoolExecutor(max_workers=2)

def _write_shard(path, data):
    torch.save(data, path)
    return path

def save_shard_async(shard_data, output_dir, shard_idx, rank):
    """Submit shard write to background thread. Returns a Future."""
    path = os.path.join(output_dir, f"rank{rank:02d}_shard_{shard_idx:05d}.pt")
    future = _save_executor.submit(_write_shard, path, shard_data)
    return future, path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute C3 encoder latents")
    parser.add_argument("--data_path",  type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="liufanfanlff/C3-Context-Cascade-Compression")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="H100 can handle 256+ comfortably")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--compile",    action="store_true",
                        help="torch.compile the encoder (~20-40%% faster after warmup)")
    parser.add_argument("--rank",       type=int, default=0,
                        help="This process's GPU rank (0-indexed)")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of parallel precompute processes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Read data ---
    print(f"Reading {args.data_path}...")
    examples = []
    with open(args.data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    n_total = len(examples)
    n_local = (n_total + args.world_size - 1) // args.world_size
    print(f"Total examples: {n_total}  |  "
          f"This rank ({args.rank}/{args.world_size}): ~{n_local}")

    # --- Load model ---
    model, tokenizer = load_model(
        args.model_name, device=args.device, compile_encoder=args.compile)
    latent_len = model.config.latent_token_len  # 32

    # --- Pre-tokenize this rank's slice ---
    print(f"\n[rank {args.rank}] Pre-tokenizing...")
    ctx_ids_list, target_ids_list, local_indices = pretokenize_all(
        tokenizer, examples, latent_len, args.rank, args.world_size)

    # Sort by context length for minimal padding waste
    order = sorted(range(len(ctx_ids_list)), key=lambda i: len(ctx_ids_list[i]))
    ctx_ids_list    = [ctx_ids_list[i]    for i in order]
    target_ids_list = [target_ids_list[i] for i in order]
    n_local = len(ctx_ids_list)
    print(f"[rank {args.rank}] {n_local} examples, sorted by length.")

    # --- Run encoder ---
    print(f"\n[rank {args.rank}] Running encoder "
          f"(batch_size={args.batch_size}, compile={args.compile})...")
    shard_buf    = []
    shard_idx    = 0
    total_done   = 0
    pending_save = None   # most recent async save Future
    t0           = time.time()

    for start in range(0, n_local, args.batch_size):
        end       = min(start + args.batch_size, n_local)
        batch_ctx = ctx_ids_list[start:end]
        batch_tgt = target_ids_list[start:end]

        latents = encode_batch(model, batch_ctx, device=args.device)

        for i in range(len(batch_ctx)):
            shard_buf.append({
                "latents":    latents[i],     # [32, 2048] bfloat16 CPU
                "target_ids": batch_tgt[i],   # [L] int64 CPU
            })
            if len(shard_buf) >= args.shard_size:
                # Wait for previous async write to finish before submitting new one
                if pending_save is not None:
                    pending_save.result()

                future, path = save_shard_async(
                    shard_buf, args.output_dir, shard_idx, args.rank)
                pending_save = future
                shard_idx  += 1
                total_done += len(shard_buf)
                elapsed     = time.time() - t0
                rate        = total_done / elapsed
                eta         = (n_local - total_done) / rate
                print(f"  [rank {args.rank}] {path} | "
                      f"{total_done}/{n_local} | "
                      f"{rate:.0f} ex/s | ETA {eta/60:.1f} min")
                shard_buf = []

    # Final shard
    if shard_buf:
        if pending_save is not None:
            pending_save.result()
        future, path = save_shard_async(
            shard_buf, args.output_dir, shard_idx, args.rank)
        future.result()
        total_done += len(shard_buf)
        print(f"  [rank {args.rank}] {path} | {total_done}/{n_local}")
    elif pending_save is not None:
        pending_save.result()

    elapsed = time.time() - t0
    print(f"\n[rank {args.rank}] Done. {total_done} examples in "
          f"{elapsed/60:.1f} min ({total_done/elapsed:.0f} ex/s avg)")


if __name__ == "__main__":
    main()
