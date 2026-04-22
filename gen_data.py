"""
gen_data.py -- Generate synthetic arithmetic and logic chain data for MTP training.

Produces JSONL files with controllable token length (100-1300 tokens via Qwen tokenizer).
500K training examples + 5K eval examples at evenly distributed lengths.

Speedups vs naive version:
  - Estimate-then-trim: generate full chain in one shot, tokenize ONCE to verify,
    trim trailing steps if over budget. Eliminates per-step tokenizer calls.
  - Multiprocessing with fork: tokenizer loaded ONCE in main process, inherited by
    all workers via copy-on-write memory — no per-worker download or disk read.
  - Batch JSONL write at the end.

Usage:
    python gen_data.py                        # defaults: 500K train, 5K eval
    python gen_data.py --num_workers 32       # tune to CPU core count
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time

# Prevent OpenBLAS / OpenMP from spawning dozens of threads per worker.
# Each gen_data worker is CPU-bound on pure Python — one thread suffices.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Disable HuggingFace tokenizer parallelism inside each worker.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Tokenizer global — loaded ONCE in main process, inherited by all workers via fork
# ---------------------------------------------------------------------------

_tokenizer = None


def _load_tokenizer(tokenizer_name):
    """Load tokenizer into the global in the MAIN process before forking.

    With fork start method, child processes inherit this directly from
    the parent's memory (copy-on-write). No per-worker download or disk read.
    """
    global _tokenizer
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True)


def _tok_len(text):
    return len(_tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Arithmetic chain — estimate then trim
# ---------------------------------------------------------------------------

# Empirical: ", 1234 + 567 = 1801" encodes to ~8 tokens on Qwen.
# Numbers with more digits encode to slightly more. We use 9 as a
# conservative over-estimate so we never fall short.
_ARITH_TOKENS_PER_STEP = 9


def _arith_step(rng, val):
    """Generate one arithmetic step. Returns (op_str, new_val)."""
    choice = rng.randint(0, 2)
    if choice == 0:
        operand = rng.randint(1, 9999)
        return f", {val} + {operand} = {val + operand}", val + operand
    elif choice == 1:
        if val <= 1:
            operand = rng.randint(1, 9999)
            return f", {val} + {operand} = {val + operand}", val + operand
        operand = rng.randint(1, min(val - 1, 9999))
        return f", {val} - {operand} = {val - operand}", val - operand
    else:
        operand = rng.randint(2, 20)
        return f", {val} * {operand} = {val * operand}", val * operand


def gen_arithmetic(rng, target_min, target_max):
    """
    Generate arithmetic chain targeting [target_min, target_max] tokens.
    Tokenizes exactly ONCE (at the end). Trims from the tail if over budget.
    """
    target = rng.randint(target_min, target_max)
    n_steps = max(1, target // _ARITH_TOKENS_PER_STEP)

    val = rng.randint(1, 9999)
    steps = [str(val)]
    for _ in range(n_steps + 10):  # small buffer in case estimate is short
        step_str, val = _arith_step(rng, val)
        steps.append(step_str)

    # Build text and trim from the tail until we're within budget
    text = "".join(steps)
    tok = _tok_len(text)

    while tok > target_max and len(steps) > 1:
        steps.pop()
        text = "".join(steps)
        tok = _tok_len(text)

    # If still too short, add more steps one at a time
    while tok < target_min:
        step_str, val = _arith_step(rng, val)
        steps.append(step_str)
        text = "".join(steps)
        tok = _tok_len(text)
        if tok > target_max:
            steps.pop()
            text = "".join(steps)
            tok = _tok_len(text)
            break

    if target_min <= tok <= target_max:
        return text, tok
    return None, 0


# ---------------------------------------------------------------------------
# Logic chain -- estimate then trim
# ---------------------------------------------------------------------------

_LOGIC_TOKENS_PER_PROP = 7   # "If A then B. " ~ 6 tokens; derivation ~ 5

PROP_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _build_logic_text(props, rng):
    """Assemble rules + premise + derivations for a list of propositions."""
    rules = []
    for i in range(len(props) - 1):
        if rng.random() < 0.2 and i > 0:
            extra = rng.choice(props[:i])
            rules.append(f"If {props[i]} and {extra} then {props[i+1]}.")
        else:
            rules.append(f"If {props[i]} then {props[i+1]}.")
    derivations = [f"Therefore {props[i]} is true." for i in range(1, len(props))]
    return " ".join(rules) + f" {props[0]} is true. " + " ".join(derivations)


def gen_logic(rng, target_min, target_max):
    """
    Generate a logic chain (or chain-of-chains for long targets).
    Tokenizes exactly ONCE. Trims or extends prop list as needed.
    """
    target = rng.randint(target_min, target_max)

    # For long targets, chain multiple independent prop groups
    n_total_props = max(4, target // _LOGIC_TOKENS_PER_PROP)

    # Build propositions: short names for first 26, then P{i}_{j} style
    def make_prop(idx):
        if idx < 26:
            return PROP_NAMES[idx]
        group, pos = divmod(idx, 26)
        return f"P{group}{PROP_NAMES[pos]}"

    props = [make_prop(i) for i in range(n_total_props)]
    rng.shuffle(props)

    text = _build_logic_text(props, rng)
    tok = _tok_len(text)

    # Trim: remove props from the tail until within budget
    while tok > target_max and len(props) > 3:
        props = props[:-1]
        text = _build_logic_text(props, rng)
        tok = _tok_len(text)

    # Extend: add props one at a time if too short
    next_idx = len(props)
    while tok < target_min:
        props.append(make_prop(next_idx))
        next_idx += 1
        text = _build_logic_text(props, rng)
        tok = _tok_len(text)
        if tok > target_max:
            props = props[:-1]
            text = _build_logic_text(props, rng)
            tok = _tok_len(text)
            break

    if target_min <= tok <= target_max:
        return text, tok
    return None, 0


# ---------------------------------------------------------------------------
# Worker function (called by multiprocessing pool)
# ---------------------------------------------------------------------------

def _generate_batch(args):
    """
    Generate a batch of examples. Called in a worker process.

    args: (seeds, min_tok, max_tok, bin_ranges)
      seeds      : list of int seeds, one per example
      min_tok    : global min (used when bin_ranges is None)
      max_tok    : global max
      bin_ranges : list of (lo, hi) per example, or None for uniform sampling
    """
    seeds, min_tok, max_tok, bin_ranges = args
    results = []

    for i, seed in enumerate(seeds):
        rng = random.Random(seed)
        lo = bin_ranges[i][0] if bin_ranges else min_tok
        hi = bin_ranges[i][1] if bin_ranges else max_tok

        dtype = rng.choice(["arithmetic", "logic"])
        if dtype == "arithmetic":
            text, tok = gen_arithmetic(rng, lo, hi)
        else:
            text, tok = gen_logic(rng, lo, hi)

        # Fallback to the other type if first failed
        if text is None:
            if dtype == "arithmetic":
                text, tok = gen_logic(rng, lo, hi)
                dtype = "logic"
            else:
                text, tok = gen_arithmetic(rng, lo, hi)
                dtype = "arithmetic"

        if text is not None and lo <= tok <= hi:
            results.append({"text": text, "type": dtype, "token_count": tok})
        else:
            results.append(None)

    return results


# ---------------------------------------------------------------------------
# Parallel generation
# ---------------------------------------------------------------------------

def generate_parallel(num_examples, min_tok, max_tok, base_seed,
                      num_workers, bin_ranges=None, chunk_size=500):
    """
    Generate num_examples examples using a multiprocessing pool.

    Tokenizer must already be loaded in the main process (_load_tokenizer called
    before this). Workers inherit it via fork — no re-loading in workers.

    bin_ranges: list of (lo, hi) per example for eval binning, or None.
    """
    master_rng = random.Random(base_seed)
    seeds = [master_rng.randint(0, 2**31) for _ in range(num_examples * 2)]

    # No initializer needed: tokenizer is already in global _tokenizer,
    # inherited by all workers via fork (copy-on-write).
    pool = mp.Pool(processes=num_workers)

    results = []
    seed_idx = 0
    t0 = time.time()
    attempts = 0

    while len(results) < num_examples:
        needed = num_examples - len(results)
        chunks = []
        bin_offset = len(results)  # track how far into bin_ranges we've assigned
        for _ in range(0, min(needed * 2, num_examples * 2 - seed_idx), chunk_size):
            if seed_idx >= len(seeds):
                seeds.extend(master_rng.randint(0, 2**31)
                             for _ in range(chunk_size))
            batch_seeds = seeds[seed_idx: seed_idx + chunk_size]
            seed_idx += chunk_size

            if bin_ranges is not None:
                batch_bins = bin_ranges[bin_offset: bin_offset + len(batch_seeds)]
                while len(batch_bins) < len(batch_seeds):
                    batch_bins.append((min_tok, max_tok))
                bin_offset += len(batch_seeds)
            else:
                batch_bins = None

            chunks.append((batch_seeds, min_tok, max_tok, batch_bins))
            attempts += len(batch_seeds)

            if len(results) + len(chunks) * chunk_size >= num_examples:
                break

        for batch_results in pool.map(_generate_batch, chunks):
            for ex in batch_results:
                if ex is not None and len(results) < num_examples:
                    results.append(ex)

        elapsed = time.time() - t0
        rate = len(results) / elapsed if elapsed > 0 else 0
        print(f"  {len(results)}/{num_examples}  ({rate:.0f} ex/s, "
              f"{attempts} attempts)")

        if attempts > num_examples * 10:
            print("  Warning: high attempt count — check token range settings.")
            break

    pool.close()
    pool.join()
    return results[:num_examples]


# ---------------------------------------------------------------------------
# Eval bin assignment
# ---------------------------------------------------------------------------

def make_eval_bin_ranges(num_examples, min_tok, max_tok, bin_width=50):
    """Assign each eval example to a length bin for uniform length distribution."""
    bins = list(range(min_tok, max_tok, bin_width))
    n_bins = len(bins)
    per_bin = max(1, num_examples // n_bins)
    remainder = num_examples - per_bin * n_bins

    ranges = []
    for i, lo in enumerate(bins):
        hi = min(lo + bin_width, max_tok)
        count = per_bin + (1 if i < remainder else 0)
        ranges.extend([(lo, hi)] * count)

    random.Random(0).shuffle(ranges)
    return ranges[:num_examples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output_train",  type=str, default="data/train.jsonl")
    parser.add_argument("--output_eval",   type=str, default="data/eval.jsonl")
    parser.add_argument("--num_train",     type=int, default=500_000)
    parser.add_argument("--num_eval",      type=int, default=5_000)
    parser.add_argument("--min_tokens",    type=int, default=100)
    parser.add_argument("--max_tokens",    type=int, default=1300)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--tokenizer",     type=str,
                        default="liufanfanlff/C3-Context-Cascade-Compression")
    parser.add_argument("--num_workers",   type=int,
                        default=max(1, mp.cpu_count() - 1),
                        help="Worker processes (default: nCPU-1)")
    parser.add_argument("--chunk_size",    type=int, default=500,
                        help="Examples per pool task")
    args = parser.parse_args()

    print(f"Workers: {args.num_workers}  |  chunk_size: {args.chunk_size}")
    print(f"Tokenizer: {args.tokenizer}")

    # Load tokenizer ONCE here in the main process.
    # With fork, all workers inherit this directly — no per-worker loading.
    print("Loading tokenizer (once, shared with all workers via fork)...")
    _load_tokenizer(args.tokenizer)
    print("Tokenizer ready.")

    # --- Training set ---
    print(f"\nGenerating {args.num_train} training examples "
          f"[{args.min_tokens}-{args.max_tokens} tokens]...")
    t0 = time.time()
    train_examples = generate_parallel(
        num_examples=args.num_train,
        min_tok=args.min_tokens,
        max_tok=args.max_tokens,
        base_seed=args.seed,
        num_workers=args.num_workers,
        bin_ranges=None,
        chunk_size=args.chunk_size,
    )
    print(f"Train generation done in {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.output_train) or ".", exist_ok=True)
    with open(args.output_train, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(train_examples)} examples to {args.output_train}")

    # --- Eval set ---
    print(f"\nGenerating {args.num_eval} eval examples "
          f"(evenly distributed across {args.min_tokens}-{args.max_tokens} tokens)...")
    eval_bin_ranges = make_eval_bin_ranges(
        args.num_eval, args.min_tokens, args.max_tokens)
    t0 = time.time()
    eval_examples = generate_parallel(
        num_examples=args.num_eval,
        min_tok=args.min_tokens,
        max_tok=args.max_tokens,
        base_seed=args.seed + 1,
        num_workers=args.num_workers,
        bin_ranges=eval_bin_ranges,
        chunk_size=args.chunk_size,
    )
    print(f"Eval generation done in {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.output_eval) or ".", exist_ok=True)
    with open(args.output_eval, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(eval_examples)} examples to {args.output_eval}")

    # --- Stats ---
    for name, exs in [("Train", train_examples), ("Eval", eval_examples)]:
        if not exs:
            continue
        lengths = [e["token_count"] for e in exs]
        n_arith = sum(1 for e in exs if e["type"] == "arithmetic")
        print(f"\n{name} stats: total={len(exs)} | "
              f"arithmetic={n_arith} logic={len(exs)-n_arith} | "
              f"len min={min(lengths)} max={max(lengths)} "
              f"mean={sum(lengths)/len(lengths):.0f}")


if __name__ == "__main__":
    # fork on Linux: workers inherit the already-loaded tokenizer from the main
    # process via copy-on-write — no re-download, no per-worker disk read.
    mp.set_start_method("fork", force=True)
    main()
