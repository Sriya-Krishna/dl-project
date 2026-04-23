"""
gen_data.py -- Generate synthetic arithmetic and logic chain data for MTP training.

Produces JSONL files with controllable token length (100-1300 tokens via Qwen tokenizer).
500K training examples + 5K eval examples at evenly distributed lengths.

Speedups:
  - Char-based generation: text is built targeting a character budget (no tokenizer
    in the hot loop). One batch-tokenize call per worker chunk verifies lengths.
  - Batch tokenization: entire chunk tokenized in a single Rust call (~5-10x faster
    than sequential Python encode calls).
  - Calibrated chars/token ratio: measured from sample texts at startup.
  - imap_unordered: results stream back as workers finish (instant progress).
  - Multiprocessing with fork: tokenizer loaded ONCE in main, inherited by workers.

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
from bisect import bisect_right

# Prevent OpenBLAS / OpenMP from spawning dozens of threads per worker.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Tokenizer global — loaded ONCE in main process, inherited by workers via fork
# ---------------------------------------------------------------------------

_tokenizer = None


def _load_tokenizer(tokenizer_name):
    global _tokenizer
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Arithmetic chain generation
# ---------------------------------------------------------------------------

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


def _gen_arith_by_chars(rng, target_chars):
    """Generate arithmetic chain targeting a character count. No tokenizer calls."""
    val = rng.randint(1, 9999)
    parts = [str(val)]
    total = len(parts[0])
    while total < target_chars:
        step_str, val = _arith_step(rng, val)
        parts.append(step_str)
        total += len(step_str)
    return "".join(parts), parts


def _trim_arith_to_chars(parts, target_chars):
    """Find largest prefix of arithmetic parts fitting within target_chars."""
    cum = []
    total = 0
    for p in parts:
        total += len(p)
        cum.append(total)
    n = bisect_right(cum, target_chars)
    return "".join(parts[:max(n, 1)])


# ---------------------------------------------------------------------------
# Logic chain generation
# ---------------------------------------------------------------------------

PROP_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _make_prop(idx):
    if idx < 26:
        return PROP_NAMES[idx]
    group, pos = divmod(idx, 26)
    return f"P{group}{PROP_NAMES[pos]}"


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


def _gen_logic_by_chars(rng, target_chars):
    """Generate logic chain targeting a character count. No tokenizer calls.

    Returns (text, props, rng_state_before_build) so the trim path can
    rebuild with fewer props using the same random conjunction pattern.
    """
    chars_per_prop = 28  # approximate average
    n_props = max(4, target_chars // chars_per_prop)
    props = [_make_prop(i) for i in range(n_props)]
    rng.shuffle(props)
    rng_state = rng.getstate()
    text = _build_logic_text(props, rng)
    # Extend if too short
    next_idx = len(props)
    while len(text) < target_chars and next_idx < 500:
        props.append(_make_prop(next_idx))
        next_idx += 1
        rng_state = rng.getstate()
        text = _build_logic_text(props, rng)
    return text, props, rng_state


def _trim_logic_to_chars(props, rng_state, target_chars):
    """Binary search for largest prop count whose text fits target_chars.

    Resetting rng_state before each _build_logic_text call ensures the
    random conjunctions for the first M-1 props are identical to the
    original text (same rng starting state, same loop iterations).
    """
    lo, hi = 3, len(props)
    best_text = None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        rng = random.Random()
        rng.setstate(rng_state)
        text = _build_logic_text(props[:mid], rng)
        if len(text) <= target_chars:
            lo = mid
            best_text = text
        else:
            hi = mid - 1
    if best_text is None:
        rng = random.Random()
        rng.setstate(rng_state)
        best_text = _build_logic_text(props[:lo], rng)
    return best_text


# ---------------------------------------------------------------------------
# Worker function — char-based generation + batch tokenize
# ---------------------------------------------------------------------------

def _generate_batch(args):
    """
    Generate a batch of examples using char estimates, then batch-tokenize.

    args: (seeds, min_tok, max_tok, bin_ranges, cpt_arith, cpt_logic)
    """
    seeds, min_tok, max_tok, bin_ranges, cpt_arith, cpt_logic = args
    n = len(seeds)

    # Phase 1: Generate texts using char estimates (no tokenizer calls)
    texts = [None] * n
    dtypes = [None] * n
    bounds = [None] * n
    arith_parts = [None] * n
    logic_trim_data = [None] * n  # (props, rng_state)

    for i, seed in enumerate(seeds):
        rng = random.Random(seed)
        lo = bin_ranges[i][0] if bin_ranges else min_tok
        hi = bin_ranges[i][1] if bin_ranges else max_tok
        bounds[i] = (lo, hi)
        target_tok = rng.randint(lo, hi)

        dtype = rng.choice(["arithmetic", "logic"])
        cpt = cpt_arith if dtype == "arithmetic" else cpt_logic
        target_chars = int(target_tok * cpt)

        if dtype == "arithmetic":
            text, parts = _gen_arith_by_chars(rng, target_chars)
            arith_parts[i] = parts
        else:
            text, props, rng_state = _gen_logic_by_chars(rng, target_chars)
            logic_trim_data[i] = (props, rng_state)

        texts[i] = text
        dtypes[i] = dtype

    # Phase 2: Batch tokenize (one Rust call for the entire chunk)
    encoded = _tokenizer(texts, add_special_tokens=False)

    # Phase 3: Check bounds, trim overshoots
    results = [None] * n
    to_retrim = []

    for i in range(n):
        lo, hi = bounds[i]
        tok = len(encoded["input_ids"][i])

        if lo <= tok <= hi:
            results[i] = {"text": texts[i], "type": dtypes[i],
                          "token_count": tok}
        elif tok > hi:
            # Overshoot: trim using per-type char estimate
            cpt = cpt_arith if dtypes[i] == "arithmetic" else cpt_logic
            target_chars = int(((lo + hi) // 2) * cpt)
            if dtypes[i] == "arithmetic" and arith_parts[i]:
                texts[i] = _trim_arith_to_chars(arith_parts[i], target_chars)
            elif dtypes[i] == "logic" and logic_trim_data[i]:
                props, rng_state = logic_trim_data[i]
                texts[i] = _trim_logic_to_chars(props, rng_state, target_chars)
            else:
                continue
            to_retrim.append(i)
        # else: undershoot — leave as None (will be retried)

    # Phase 4: Re-verify trimmed texts (second batch tokenize)
    if to_retrim:
        retrim_texts = [texts[i] for i in to_retrim]
        re_enc = _tokenizer(retrim_texts, add_special_tokens=False)
        for j, idx in enumerate(to_retrim):
            lo, hi = bounds[idx]
            tok = len(re_enc["input_ids"][j])
            if lo <= tok <= hi:
                results[idx] = {"text": texts[idx], "type": dtypes[idx],
                                "token_count": tok}

    return results


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _calibrate_cpt():
    """Compute per-type chars-per-token ratios from sample texts.

    Returns (cpt_arith, cpt_logic) — separate ratios because arithmetic
    (number-heavy, ~1.1 c/t) and logic (English words, ~3+ c/t) diverge
    heavily. A blended average causes massive under/overshoot.
    """
    rng = random.Random(12345)
    arith_samples = []
    logic_samples = []

    for _ in range(50):
        val = rng.randint(1, 9999)
        chain = str(val)
        for _ in range(rng.randint(10, 100)):
            step, val = _arith_step(rng, val)
            chain += step
        arith_samples.append(chain)

    for _ in range(50):
        n = rng.randint(5, 50)
        props = [_make_prop(i) for i in range(n)]
        rng.shuffle(props)
        text = _build_logic_text(props, rng)
        logic_samples.append(text)

    arith_enc = _tokenizer(arith_samples, add_special_tokens=False)
    logic_enc = _tokenizer(logic_samples, add_special_tokens=False)

    arith_chars = sum(len(s) for s in arith_samples)
    arith_tokens = sum(len(ids) for ids in arith_enc["input_ids"])
    cpt_arith = arith_chars / arith_tokens

    logic_chars = sum(len(s) for s in logic_samples)
    logic_tokens = sum(len(ids) for ids in logic_enc["input_ids"])
    cpt_logic = logic_chars / logic_tokens

    print(f"  Calibrated chars/token: arithmetic={cpt_arith:.3f}, "
          f"logic={cpt_logic:.3f}")
    return cpt_arith, cpt_logic


# ---------------------------------------------------------------------------
# Parallel generation with streaming progress
# ---------------------------------------------------------------------------

def generate_parallel(num_examples, min_tok, max_tok, base_seed,
                      num_workers, cpt_arith, cpt_logic,
                      bin_ranges=None, chunk_size=500):
    """Generate num_examples using multiprocessing with streaming progress."""
    master_rng = random.Random(base_seed)

    def chunks():
        bin_offset = 0
        yielded = 0
        limit = num_examples * 3  # safety cap
        while yielded < limit:
            batch_seeds = [master_rng.randint(0, 2**31)
                           for _ in range(chunk_size)]
            if bin_ranges is not None:
                batch_bins = bin_ranges[bin_offset:bin_offset + chunk_size]
                while len(batch_bins) < chunk_size:
                    batch_bins.append((min_tok, max_tok))
                bin_offset += chunk_size
            else:
                batch_bins = None
            yield (batch_seeds, min_tok, max_tok, batch_bins,
                   cpt_arith, cpt_logic)
            yielded += chunk_size

    pool = mp.Pool(processes=num_workers)
    results = []
    t0 = time.time()
    last_print = 0

    for batch_results in pool.imap_unordered(_generate_batch, chunks()):
        for ex in batch_results:
            if ex is not None and len(results) < num_examples:
                results.append(ex)

        done = len(results)
        if done - last_print >= 10000 or done >= num_examples:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_examples - done) / rate if rate > 0 else 0
            print(f"  {done}/{num_examples}  ({rate:.0f} ex/s, ETA {eta:.0f}s)")
            last_print = done

        if done >= num_examples:
            break

    pool.terminate()
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

    print("Loading tokenizer (once, shared with all workers via fork)...")
    _load_tokenizer(args.tokenizer)
    print("Tokenizer ready.")

    print("Calibrating chars/token ratio...")
    cpt_arith, cpt_logic = _calibrate_cpt()

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
        cpt_arith=cpt_arith,
        cpt_logic=cpt_logic,
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
        cpt_arith=cpt_arith,
        cpt_logic=cpt_logic,
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
    mp.set_start_method("fork", force=True)
    main()
