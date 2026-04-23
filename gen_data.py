"""
gen_data.py -- Prepare real text data for MTP training.

Downloads text from HuggingFace, tokenizes with the Qwen tokenizer,
chunks into segments of controllable token length, and saves as JSONL.

Default dataset: wikitext-103-raw-v1 (~100M tokens, yields ~100-150K examples).
For more data, try: --dataset Skylion007/openwebtext (no config needed).

Usage:
    python gen_data.py                                    # wikitext-103 defaults
    python gen_data.py --num_train 50000 --num_eval 2000  # smaller run
    python gen_data.py --dataset Skylion007/openwebtext --dataset_config none
"""

import argparse
import json
import os
import random
import time

from datasets import load_dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_token_ids(all_ids, min_tok, max_tok, rng):
    """Chunk a long token ID sequence into segments of random length.

    Each segment is [min_tok, max_tok] tokens. Segments shorter than
    min_tok at the tail are discarded.
    """
    chunks = []
    i = 0
    while i < len(all_ids):
        target = rng.randint(min_tok, max_tok)
        chunk = all_ids[i:i + target]
        if len(chunk) >= min_tok:
            chunks.append(chunk)
        i += target
    return chunks


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_texts(dataset_name, dataset_config, text_field, splits):
    """Load raw text from a HuggingFace dataset. Returns list of strings."""
    print(f"Loading dataset: {dataset_name}"
          f"{f' ({dataset_config})' if dataset_config else ''}...")

    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config)
    else:
        ds = load_dataset(dataset_name)

    texts = []
    for split in splits:
        if split not in ds:
            print(f"  Split '{split}' not found, skipping.")
            continue
        for example in ds[split]:
            text = example[text_field].strip()
            if len(text) >= 50:
                texts.append(text)
        print(f"  Loaded {split}: {len(texts)} texts so far")

    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare real text data for MTP training")
    parser.add_argument("--output_train", type=str, default="data_real/train.jsonl")
    parser.add_argument("--output_eval",  type=str, default="data_real/eval.jsonl")
    parser.add_argument("--num_train",    type=int, default=100_000)
    parser.add_argument("--num_eval",     type=int, default=2_000)
    parser.add_argument("--min_tokens",   type=int, default=100)
    parser.add_argument("--max_tokens",   type=int, default=1300)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--tokenizer",    type=str,
                        default="liufanfanlff/C3-Context-Cascade-Compression")
    parser.add_argument("--dataset",      type=str, default="wikitext",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str,
                        default="wikitext-103-raw-v1",
                        help="Dataset config (use 'none' to skip)")
    parser.add_argument("--text_field",   type=str, default="text",
                        help="Field name containing text in the dataset")
    parser.add_argument("--splits",       type=str, default="train,validation,test",
                        help="Comma-separated splits to use")
    args = parser.parse_args()

    if args.dataset_config == "none":
        args.dataset_config = None

    rng = random.Random(args.seed)

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True)

    # --- Load raw texts ---
    splits = [s.strip() for s in args.splits.split(",")]
    texts = load_texts(args.dataset, args.dataset_config,
                       args.text_field, splits)
    print(f"Total texts loaded: {len(texts)}")

    # --- Tokenize and chunk ---
    print(f"\nTokenizing and chunking to [{args.min_tokens}-{args.max_tokens}] tokens...")
    t0 = time.time()
    all_chunks = []

    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < args.min_tokens:
            continue

        if len(ids) <= args.max_tokens:
            # Fits in one chunk
            all_chunks.append(ids)
        else:
            # Split long texts into multiple chunks
            chunks = chunk_token_ids(ids, args.min_tokens, args.max_tokens, rng)
            all_chunks.extend(chunks)

        if (i + 1) % 50_000 == 0:
            print(f"  Processed {i+1}/{len(texts)} texts, "
                  f"{len(all_chunks)} chunks so far ({time.time()-t0:.0f}s)")

    print(f"Total chunks: {len(all_chunks)} ({time.time()-t0:.0f}s)")

    # --- Shuffle and split ---
    rng.shuffle(all_chunks)

    total_needed = args.num_train + args.num_eval
    if len(all_chunks) < total_needed:
        print(f"Warning: only {len(all_chunks)} chunks available, "
              f"need {total_needed}. Using all available.")
        args.num_eval = min(args.num_eval, len(all_chunks) // 10)
        args.num_train = len(all_chunks) - args.num_eval

    eval_chunks = all_chunks[:args.num_eval]
    train_chunks = all_chunks[args.num_eval:args.num_eval + args.num_train]

    # --- Decode and write JSONL ---
    for name, chunks, path in [("Train", train_chunks, args.output_train),
                                ("Eval",  eval_chunks,  args.output_eval)]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for chunk_ids in chunks:
                text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                f.write(json.dumps({
                    "text": text,
                    "type": "real",
                    "token_count": len(chunk_ids),
                }) + "\n")

        lengths = [len(c) for c in chunks]
        print(f"\n{name}: {len(chunks)} examples → {path}")
        if lengths:
            print(f"  tokens: min={min(lengths)} max={max(lengths)} "
                  f"mean={sum(lengths)/len(lengths):.0f}")


if __name__ == "__main__":
    main()
