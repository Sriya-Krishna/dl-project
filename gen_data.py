"""
gen_data.py — Generate synthetic arithmetic and logic chain data for MTP training.

Produces JSONL files with controllable token length (100-1300 tokens via Qwen tokenizer).
500K training examples + 5K eval examples at evenly distributed lengths.
"""

import argparse
import json
import os
import random
from pathlib import Path

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Arithmetic chain generator
# ---------------------------------------------------------------------------

def gen_arithmetic_chain(rng, tokenizer, min_tok, max_tok):
    """Build an arithmetic chain like '234 + 567 = 801, 801 * 2 = 1602, ...'"""
    ops = ["+", "-", "*"]
    val = rng.randint(1, 9999)
    parts = [str(val)]
    text = str(val)

    for _ in range(500):  # upper bound on chain length
        op = rng.choice(ops)
        if op == "+":
            operand = rng.randint(1, 9999)
            result = val + operand
        elif op == "-":
            if val <= 1:
                # Can't subtract; fall through to addition
                op = "+"
                operand = rng.randint(1, 9999)
                result = val + operand
            else:
                operand = rng.randint(1, min(val - 1, 9999))  # keep result >= 1
                result = val - operand
        else:  # *
            operand = rng.randint(2, 20)  # keep numbers manageable
            result = val * operand

        step = f", {val} {op} {operand} = {result}"
        candidate = text + step
        tok_len = len(tokenizer.encode(candidate))

        if tok_len > max_tok:
            break
        text = candidate
        val = result

        if tok_len >= min_tok:
            # Randomly decide to stop (30% chance) once we're in range
            if rng.random() < 0.3:
                break

    return text


def gen_arithmetic_chain_targeted(rng, tokenizer, target_min, target_max,
                                  max_retries=20):
    """Generate an arithmetic chain within a specific token length range."""
    for _ in range(max_retries):
        text = gen_arithmetic_chain(rng, tokenizer, target_min, target_max)
        tok_len = len(tokenizer.encode(text))
        if target_min <= tok_len <= target_max:
            return text, tok_len
    return None, 0


# ---------------------------------------------------------------------------
# Logic chain generator
# ---------------------------------------------------------------------------

PROP_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def gen_logic_chain(rng, tokenizer, min_tok, max_tok):
    """Build a logic chain like 'If A then B. If B then C. A is true. Therefore C is true.'"""
    n_props = rng.randint(3, min(15, len(PROP_NAMES)))
    props = rng.sample(PROP_NAMES, n_props)

    # Build implication chain: props[0]->props[1]->...->props[-1]
    rules = []
    for i in range(len(props) - 1):
        # Occasionally add compound conditions
        if rng.random() < 0.2 and i > 0:
            extra = rng.choice(props[:i])
            rules.append(f"If {props[i]} and {extra} then {props[i+1]}.")
        else:
            rules.append(f"If {props[i]} then {props[i+1]}.")

    # Premise
    premise = f"{props[0]} is true."

    # Derivations
    derivations = []
    for i in range(1, len(props)):
        derivations.append(f"Therefore {props[i]} is true.")

    text = " ".join(rules) + " " + premise
    for d in derivations:
        candidate = text + " " + d
        tok_len = len(tokenizer.encode(candidate))
        if tok_len > max_tok:
            break
        text = candidate
        if tok_len >= min_tok and rng.random() < 0.3:
            break

    return text


def gen_logic_chain_targeted(rng, tokenizer, target_min, target_max,
                             max_retries=20):
    """Generate a logic chain within a specific token length range."""
    for _ in range(max_retries):
        text = gen_logic_chain(rng, tokenizer, target_min, target_max)
        tok_len = len(tokenizer.encode(text))
        if target_min <= tok_len <= target_max:
            return text, tok_len
    return None, 0


# ---------------------------------------------------------------------------
# Extended logic chain for longer sequences
# ---------------------------------------------------------------------------

def gen_extended_logic_chain(rng, tokenizer, min_tok, max_tok):
    """Generate multiple logic chain segments to reach longer token counts."""
    segments = []
    total_text = ""

    for seg_idx in range(50):  # up to 50 segments
        n_props = rng.randint(4, 10)
        base_names = [f"{c}{seg_idx}" if seg_idx > 0 else c
                      for c in rng.sample(PROP_NAMES, min(n_props, len(PROP_NAMES)))]
        # Ensure unique names
        props = [f"P{seg_idx}_{i}" for i in range(n_props)] if seg_idx > 3 else base_names

        rules = []
        for i in range(len(props) - 1):
            rules.append(f"If {props[i]} then {props[i+1]}.")

        premise = f"{props[0]} is true."
        derivations = [f"Therefore {props[i]} is true." for i in range(1, len(props))]

        segment = " ".join(rules) + " " + premise + " " + " ".join(derivations)
        candidate = (total_text + " " + segment).strip()
        tok_len = len(tokenizer.encode(candidate))

        if tok_len > max_tok:
            break
        total_text = candidate
        if tok_len >= min_tok and rng.random() < 0.3:
            break

    return total_text


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_example(rng, tokenizer, min_tok, max_tok):
    """Generate a single example within the token length range."""
    dtype = rng.choice(["arithmetic", "logic"])

    if dtype == "arithmetic":
        text, tok_len = gen_arithmetic_chain_targeted(
            rng, tokenizer, min_tok, max_tok)
        if text is None:
            # Fallback to logic
            dtype = "logic"

    if dtype == "logic":
        text, tok_len = gen_logic_chain_targeted(
            rng, tokenizer, min_tok, max_tok)
        if text is None:
            # Try extended logic for longer targets
            text = gen_extended_logic_chain(rng, tokenizer, min_tok, max_tok)
            tok_len = len(tokenizer.encode(text))
            if not (min_tok <= tok_len <= max_tok):
                return None

    return {"text": text, "type": dtype, "token_count": tok_len}


def generate_train_set(rng, tokenizer, num_examples, min_tok, max_tok):
    """Generate training examples with token lengths distributed across the range."""
    examples = []
    attempts = 0
    max_attempts = num_examples * 5

    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1
        ex = generate_example(rng, tokenizer, min_tok, max_tok)
        if ex is not None:
            examples.append(ex)
            if len(examples) % 10000 == 0:
                print(f"  Generated {len(examples)}/{num_examples} train examples "
                      f"({attempts} attempts)")

    return examples


def generate_eval_set(rng, tokenizer, num_examples, min_tok, max_tok):
    """Generate eval examples with lengths evenly distributed across bins."""
    bin_width = 50
    bins = list(range(min_tok, max_tok, bin_width))
    examples_per_bin = max(1, num_examples // len(bins))
    remainder = num_examples - examples_per_bin * len(bins)

    examples = []
    for i, bin_start in enumerate(bins):
        bin_end = min(bin_start + bin_width, max_tok)
        target_count = examples_per_bin + (1 if i < remainder else 0)
        bin_examples = []
        attempts = 0

        while len(bin_examples) < target_count and attempts < target_count * 20:
            attempts += 1
            ex = generate_example(rng, tokenizer, bin_start, bin_end)
            if ex is not None:
                bin_examples.append(ex)

        examples.extend(bin_examples)
        if (i + 1) % 5 == 0:
            print(f"  Eval bins completed: {i + 1}/{len(bins)}, "
                  f"total examples: {len(examples)}")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output_train", type=str, default="data/train.jsonl")
    parser.add_argument("--output_eval", type=str, default="data/eval.jsonl")
    parser.add_argument("--num_train", type=int, default=500_000)
    parser.add_argument("--num_eval", type=int, default=5_000)
    parser.add_argument("--min_tokens", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=1300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str,
                        default="liufanfanlff/C3-Context-Cascade-Compression")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True)

    rng = random.Random(args.seed)

    # --- Training set ---
    print(f"\nGenerating {args.num_train} training examples "
          f"[{args.min_tokens}-{args.max_tokens} tokens]...")
    train_examples = generate_train_set(
        rng, tokenizer, args.num_train, args.min_tokens, args.max_tokens)

    os.makedirs(os.path.dirname(args.output_train) or ".", exist_ok=True)
    with open(args.output_train, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(train_examples)} training examples to {args.output_train}")

    # --- Eval set ---
    print(f"\nGenerating {args.num_eval} eval examples "
          f"(evenly distributed across length bins)...")
    eval_examples = generate_eval_set(
        rng, tokenizer, args.num_eval, args.min_tokens, args.max_tokens)

    os.makedirs(os.path.dirname(args.output_eval) or ".", exist_ok=True)
    with open(args.output_eval, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(eval_examples)} eval examples to {args.output_eval}")

    # --- Quick stats ---
    for name, exs in [("Train", train_examples), ("Eval", eval_examples)]:
        if not exs:
            continue
        lengths = [e["token_count"] for e in exs]
        types = [e["type"] for e in exs]
        n_arith = sum(1 for t in types if t == "arithmetic")
        n_logic = sum(1 for t in types if t == "logic")
        print(f"\n{name} stats:")
        print(f"  Total: {len(exs)}")
        print(f"  Arithmetic: {n_arith}, Logic: {n_logic}")
        print(f"  Token length — min: {min(lengths)}, max: {max(lengths)}, "
              f"mean: {sum(lengths)/len(lengths):.0f}")


if __name__ == "__main__":
    main()
