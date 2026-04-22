"""
precompute_latents.py — Run the C3 encoder on all training/eval data and save
projected latent vectors as .pt shards.

Speedups vs naive version:
  - Pre-tokenize ALL examples upfront (no tokenizer in the GPU hot loop)
  - Sort by context length for tight batching (minimal padding waste)
  - Reload encoder with flash_attention_2 after main model load
  - torch.inference_mode() instead of no_grad

For each example:
  1. Tokenize: {text_tokens} + [<img>, <imgpad>*32, </img>]
  2. Embed via llm1, replace <imgpad> positions with Q.weight
  3. Forward through llm1 -> extract hidden_states[-1] at query positions -> [32, 1536]
  4. Project via mm_projector -> [32, 2048]
  5. Save {latents: [32,2048], target_ids: [L]} per example
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name, device="cuda"):
    """Load C3 model, then swap the encoder for a flash_attention_2 version."""
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

    # Swap encoder (llm1) for flash_attention_2 version — allows larger batches
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
    print("Model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Pre-tokenize all data upfront (CPU, done once)
# ---------------------------------------------------------------------------

def pretokenize_all(tokenizer, examples, latent_len, chunk_size=2000):
    """
    Tokenize every example once on CPU before any GPU work.

    Context IDs = raw text token IDs + [<img>, <imgpad>*32, </img>]
    We append the suffix token IDs directly instead of string-building,
    which is ~4x faster for large datasets.

    Returns:
        ctx_ids_list  : list of lists of int   (variable length, no padding)
        target_ids_list: list of torch.LongTensor
    """
    im_start = tokenizer.convert_tokens_to_ids("<img>")
    im_patch = tokenizer.convert_tokens_to_ids("<imgpad>")
    im_end   = tokenizer.convert_tokens_to_ids("</img>")
    suffix   = [im_start] + [im_patch] * latent_len + [im_end]

    texts = [ex["text"] for ex in examples]
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
            elapsed = time.time() - t0
            print(f"  Pre-tokenized {done}/{len(texts)}  ({elapsed:.0f}s)")

    return ctx_ids_list, target_ids_list


# ---------------------------------------------------------------------------
# Encoder batch forward (operates on pre-tokenized IDs)
# ---------------------------------------------------------------------------

def encode_batch(model, ctx_ids_batch, device="cuda"):
    """
    Run one GPU batch.

    Args:
        ctx_ids_batch: list of pre-tokenized context ID lists (variable len)

    Returns:
        latents_proj: [B, 32, 2048]  bfloat16  on CPU
    """
    config     = model.config
    latent_len = config.latent_token_len   # 32
    im_start   = config.im_start_token     # 151857
    encoder    = model.model.llm1
    Q_weight   = model.model.Q.weight      # [32, 1536]
    mm_proj    = model.model.mm_projector  # Linear(1536, 2048)

    # Pad batch to uniform length
    lengths  = [len(ids) for ids in ctx_ids_batch]
    max_len  = max(lengths)
    pad_id   = 0  # padding doesn't matter — masked out
    B        = len(ctx_ids_batch)

    input_ids_np  = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, (ids, L) in enumerate(zip(ctx_ids_batch, lengths)):
        input_ids_np[i, :L]    = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, :L]  = 1

    # Find <img> positions (always right-aligned before </imgpad>s)
    # <img> is the (L - latent_len - 2)-th token (0-indexed) for each example,
    # but we verify via nonzero for safety.
    img_start_positions = []
    for i in range(B):
        hits = (input_ids_np[i] == im_start).nonzero(as_tuple=True)[0]
        assert len(hits) == 1, f"Example {i}: expected 1 <img>, got {len(hits)}"
        img_start_positions.append(hits[0].item())

    # Embed and inject Q.weight
    context_embeds = encoder.model.embed_tokens(input_ids_np)  # [B, S, 1536]
    Q_cast = Q_weight.to(device=device, dtype=context_embeds.dtype)
    for i, pos in enumerate(img_start_positions):
        context_embeds[i, pos + 1: pos + 1 + latent_len] = Q_cast

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

    # Extract query positions
    latents_raw = torch.stack([
        last_hidden[i, img_start_positions[i] + 1:
                       img_start_positions[i] + 1 + latent_len]
        for i in range(B)
    ])  # [B, 32, 1536]

    # Cast to mm_projector dtype (main model may differ from llm1 output dtype)
    latents_raw = latents_raw.to(dtype=mm_proj.weight.dtype)

    # Project [B, 32, 1536] -> [B, 32, 2048]
    with torch.inference_mode():
        latents_proj = mm_proj(latents_raw)

    return latents_proj.cpu()


# ---------------------------------------------------------------------------
# Shard saving
# ---------------------------------------------------------------------------

def save_shard(shard_data, output_dir, shard_idx):
    path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pt")
    torch.save(shard_data, path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute C3 encoder latents for all examples")
    parser.add_argument("--data_path",   type=str, required=True)
    parser.add_argument("--model_name",  type=str,
                        default="liufanfanlff/C3-Context-Cascade-Compression")
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--shard_size",  type=int, default=1000)
    parser.add_argument("--batch_size",  type=int, default=128,
                        help="GPU encoder batch size. H100 can handle 128-256.")
    parser.add_argument("--device",      type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Read data ---
    print(f"Reading {args.data_path}...")
    examples = []
    with open(args.data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples.")

    # --- Load model ---
    model, tokenizer = load_model(args.model_name, device=args.device)
    latent_len = model.config.latent_token_len  # 32

    # --- Pre-tokenize all data (CPU, done once) ---
    print("\nPre-tokenizing all data...")
    ctx_ids_list, target_ids_list = pretokenize_all(
        tokenizer, examples, latent_len)

    # --- Sort by context length for minimal padding waste ---
    order = sorted(range(len(examples)), key=lambda i: len(ctx_ids_list[i]))
    ctx_ids_list    = [ctx_ids_list[i]    for i in order]
    target_ids_list = [target_ids_list[i] for i in order]
    print("Sorted by length.")

    # --- Run encoder in batches ---
    print(f"\nRunning encoder (batch_size={args.batch_size})...")
    shard_buf      = []
    shard_idx      = 0
    total_done     = 0
    t_gpu_start    = time.time()

    for start in range(0, len(examples), args.batch_size):
        end   = min(start + args.batch_size, len(examples))
        batch_ctx = ctx_ids_list[start:end]
        batch_tgt = target_ids_list[start:end]

        latents = encode_batch(model, batch_ctx, device=args.device)  # [B, 32, 2048]

        for i in range(len(batch_ctx)):
            shard_buf.append({
                "latents":    latents[i],       # [32, 2048] bfloat16, CPU
                "target_ids": batch_tgt[i],     # [L] int64, CPU
            })
            if len(shard_buf) >= args.shard_size:
                path = save_shard(shard_buf, args.output_dir, shard_idx)
                shard_idx  += 1
                total_done += len(shard_buf)
                elapsed     = time.time() - t_gpu_start
                rate        = total_done / elapsed
                eta         = (len(examples) - total_done) / rate
                print(f"  {path}  |  {total_done}/{len(examples)}"
                      f"  |  {rate:.0f} ex/s  |  ETA {eta/60:.1f} min")
                shard_buf = []

    if shard_buf:
        path = save_shard(shard_buf, args.output_dir, shard_idx)
        total_done += len(shard_buf)
        print(f"  {path}  |  {total_done}/{len(examples)}")

    elapsed = time.time() - t_gpu_start
    print(f"\nDone. {total_done} examples in {elapsed/60:.1f} min "
          f"({total_done/elapsed:.0f} ex/s avg)  ->  {args.output_dir}")


if __name__ == "__main__":
    main()
