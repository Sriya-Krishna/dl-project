"""
find_batch_size.py -- Find the largest encoder batch size that fits on your GPU.

Tests increasing batch sizes at worst-case sequence length (1334 tokens).
The last batch size that prints OK is your maximum; use 80-90% of it as
--batch_size in precompute_latents.py to leave headroom.

Usage:
    python find_batch_size.py
    python find_batch_size.py --model_name liufanfanlff/C3-Context-Cascade-Compression
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="liufanfanlff/C3-Context-Cascade-Compression")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seq_len", type=int, default=1334,
                        help="Worst-case sequence length (1300 tokens + 34 special)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    encoder_fa = Qwen2ForCausalLM.from_pretrained(
        args.model_name,
        subfolder="llm1",
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
        use_safetensors=True,
    )
    model.model.llm1 = encoder_fa
    del encoder_fa
    model.eval()
    model.requires_grad_(False)

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU: {torch.cuda.get_device_name(0)}  |  Total VRAM: {total_mem:.1f} GB")
    print(f"Seq length: {args.seq_len}  (worst case)\n")
    print(f"{'batch':>6}  {'peak_mem':>10}  status")
    print("-" * 32)

    batch_sizes = [64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024]
    last_ok = None

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            ids    = torch.zeros((bs, args.seq_len), dtype=torch.long, device=args.device)
            mask   = torch.ones((bs, args.seq_len),  dtype=torch.long, device=args.device)
            embeds = model.model.llm1.model.embed_tokens(ids)
            with torch.inference_mode():
                model.model.llm1(
                    input_ids=None,
                    inputs_embeds=embeds,
                    attention_mask=mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"{bs:>6}  {peak:>8.1f} GB  OK")
            last_ok = bs
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6}  {'':>10}  OOM")
            break

    print()
    if last_ok:
        safe = int(last_ok * 0.85)
        print(f"Max batch that fit:       {last_ok}")
        print(f"Recommended --batch_size: {safe}  (85% of max for safety)")
        print(f"\nRun precompute with:")
        print(f"  python precompute_latents.py "
              f"--data_path data/train.jsonl "
              f"--output_dir data/latents_train/ "
              f"--batch_size {safe} --compile")
    else:
        print("Even batch=64 OOM'd. Try a smaller --seq_len or check VRAM usage.")


if __name__ == "__main__":
    main()
