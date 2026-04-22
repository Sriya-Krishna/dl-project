"""
precompute_latents.py — Run the C3 encoder on all training/eval data and save
projected latent vectors as .pt shards.

For each example:
  1. Tokenize: {text}<img><imgpad>*32</img>
  2. Embed via llm1, replace <imgpad> positions with Q.weight
  3. Forward through llm1, extract hidden_states[-1] at query positions -> [32, 1536]
  4. Project via mm_projector -> [32, 2048]
  5. Save {latents: [32,2048], target_ids: [L]} per example
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"


def load_model(model_name, device="cuda"):
    """Load the full C3 model (encoder + decoder)."""
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
    model.eval()
    model.requires_grad_(False)
    print("Model loaded.")
    return model, tokenizer


def build_context_string(text, latent_len=32):
    """Build the encoder input string: {text}<img><imgpad>x32</img>"""
    pad_tokens = DEFAULT_IMAGE_PATCH_TOKEN * latent_len
    return text + DEFAULT_IM_START_TOKEN + pad_tokens + DEFAULT_IM_END_TOKEN


def encode_batch(model, tokenizer, texts, device="cuda"):
    """
    Run the C3 encoder on a batch of texts.
    Returns projected latents [B, 32, 2048] and per-example target token IDs.
    """
    config = model.config
    latent_len = config.latent_token_len  # 32
    im_start_token = config.im_start_token  # 151857

    # Build context strings and tokenize for encoder
    context_strings = [build_context_string(t, latent_len) for t in texts]
    ctx_enc = tokenizer(
        context_strings,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    context_ids = ctx_enc["input_ids"].to(device)
    context_attention_mask = ctx_enc["attention_mask"].to(device)

    # Also tokenize raw text for target token IDs
    raw_enc = tokenizer(texts, truncation=False)
    target_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in raw_enc["input_ids"]]

    # --- Encoder forward (replicates C3QwenModel.forward encoder path) ---
    encoder = model.model.llm1  # Qwen2ForCausalLM (1.5B)
    Q_weight = model.model.Q.weight  # [32, 1536]
    mm_proj = model.model.mm_projector  # Linear(1536, 2048)

    # Embed context_ids through encoder's embedding table
    context_embeds = encoder.model.embed_tokens(context_ids)  # [B, S, 1536]

    # For each example, find <img> position and replace next 32 <imgpad> with Q.weight
    batch_size = context_ids.shape[0]
    image_start_positions = []
    for i in range(batch_size):
        img_starts = (context_ids[i] == im_start_token).nonzero(as_tuple=True)[0]
        assert len(img_starts) == 1, f"Expected 1 <img> token, got {len(img_starts)}"
        pos = img_starts[0].item()
        image_start_positions.append(pos)
        # Replace positions [pos+1, pos+32] with Q.weight
        context_embeds[i, pos + 1: pos + 1 + latent_len] = Q_weight.to(
            device=context_embeds.device, dtype=context_embeds.dtype)

    # Forward through llm1
    with torch.no_grad():
        llm1_out = encoder(
            input_ids=None,
            inputs_embeds=context_embeds,
            attention_mask=context_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    last_hidden = llm1_out["hidden_states"][-1]  # [B, S, 1536]

    # Extract the 32 query positions from each example
    latents_raw = []
    for i in range(batch_size):
        pos = image_start_positions[i]
        h = last_hidden[i, pos + 1: pos + 1 + latent_len]  # [32, 1536]
        latents_raw.append(h)
    latents_raw = torch.stack(latents_raw)  # [B, 32, 1536]

    # Project to decoder space
    with torch.no_grad():
        latents_proj = mm_proj(latents_raw)  # [B, 32, 2048]

    return latents_proj, target_ids_list


def save_shard(shard_data, output_dir, shard_idx):
    """Save a shard as a .pt file."""
    path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pt")
    torch.save(shard_data, path)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Precompute C3 encoder latents for all examples")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSONL data file")
    parser.add_argument("--model_name", type=str,
                        default="liufanfanlff/C3-Context-Cascade-Compression")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save .pt shards")
    parser.add_argument("--shard_size", type=int, default=1000,
                        help="Examples per shard file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Encoder batch size")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(args.model_name, device=args.device)

    # Read all examples
    print(f"Reading data from {args.data_path}...")
    examples = []
    with open(args.data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples.")

    # Process in batches, save in shards
    shard_data = []
    shard_idx = 0
    total_processed = 0

    for batch_start in range(0, len(examples), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(examples))
        batch_texts = [ex["text"] for ex in examples[batch_start:batch_end]]

        latents_proj, target_ids_list = encode_batch(
            model, tokenizer, batch_texts, device=args.device)

        for i in range(len(batch_texts)):
            shard_data.append({
                "latents": latents_proj[i].cpu(),       # [32, 2048] bf16
                "target_ids": target_ids_list[i].cpu(), # [L] int64
            })

            if len(shard_data) >= args.shard_size:
                path = save_shard(shard_data, args.output_dir, shard_idx)
                shard_idx += 1
                total_processed += len(shard_data)
                print(f"  Saved {path} ({total_processed}/{len(examples)} processed)")
                shard_data = []

    # Save remaining
    if shard_data:
        path = save_shard(shard_data, args.output_dir, shard_idx)
        total_processed += len(shard_data)
        print(f"  Saved {path} ({total_processed}/{len(examples)} processed)")

    print(f"\nDone. {total_processed} examples saved across {shard_idx + 1} shards "
          f"in {args.output_dir}")


if __name__ == "__main__":
    main()
