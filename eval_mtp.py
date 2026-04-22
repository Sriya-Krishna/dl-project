"""
eval_mtp.py -- Run evaluation on a trained MTP checkpoint against the held-out set.

Metrics:
  - Exact match (boolean)
  - Character-level edit distance
  - Token-level edit distance
  - MTP draft acceptance rate per offset k
Results grouped by input token length, output as CSV.
"""

import argparse
import csv
import glob
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from peft import PeftModel

try:
    import Levenshtein
except ImportError:
    Levenshtein = None


# ---------------------------------------------------------------------------
# MTP Head (must match training definition)
# ---------------------------------------------------------------------------

class MTPHead(nn.Module):
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


# ---------------------------------------------------------------------------
# Edit distance utilities
# ---------------------------------------------------------------------------

def char_edit_distance(s1, s2):
    if Levenshtein is not None:
        return Levenshtein.distance(s1, s2)
    # Fallback: simple DP
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def token_edit_distance(tokens1, tokens2):
    """Levenshtein distance on token ID sequences."""
    m, n = len(tokens1), len(tokens2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# Load model + checkpoint
# ---------------------------------------------------------------------------

def load_model_for_evaluation(model_name, checkpoint_dir, device="cuda"):
    """Load base C3 decoder + LoRA adapter + MTP head from checkpoint."""
    print(f"Loading base model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device,
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Free encoder
    base_model.model.llm1 = None
    base_model.model.Q = None
    base_model.model.mm_projector = None
    torch.cuda.empty_cache()

    # Load LoRA adapter
    lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
    print(f"Loading LoRA adapter from {lora_dir}...")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.requires_grad_(False)

    # Load MTP head
    hidden_dim = model.base_model.model.config.hidden_size
    mtp_head = MTPHead(hidden_dim=hidden_dim).to(device=device, dtype=torch.bfloat16)
    mtp_path = os.path.join(checkpoint_dir, "mtp_head.pt")
    print(f"Loading MTP head from {mtp_path}...")
    mtp_head.load_state_dict(
        torch.load(mtp_path, map_location=device, weights_only=True))
    mtp_head.requires_grad_(False)

    return model, mtp_head, tokenizer


# ---------------------------------------------------------------------------
# Load data from shards
# ---------------------------------------------------------------------------

def load_shard_data(latent_dir):
    """Load all shards into a flat list."""
    shard_paths = sorted(
        glob.glob(os.path.join(latent_dir, "shard_*.pt")) +
        glob.glob(os.path.join(latent_dir, "rank*_shard_*.pt"))
    )
    assert shard_paths, f"No shards found in {latent_dir}"
    data = []
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu", weights_only=True)
        data.extend(shard)
    print(f"Loaded {len(data)} examples from {len(shard_paths)} shards.")
    return data


# ---------------------------------------------------------------------------
# Generation with MTP drafts
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_mtp(model, mtp_head, latent_embeds, prompt_embeds,
                      max_new_tokens, K, eos_id, device):
    """
    Custom autoregressive generation with KV cache.
    Also records MTP draft predictions at each step for acceptance rate.

    Returns:
        generated: list of token IDs
        mtp_drafts: list of K-length lists of draft token IDs per step
    """
    base = model.base_model.model          # C3QwenForCausalLM
    lm_head = base.lm_head
    embed_fn = base.model.embed_tokens
    backbone = base.model                   # C3QwenModel (LoRA active)

    # Prefill: forward full prefix [latents | prompt]
    prefix = torch.cat([latent_embeds, prompt_embeds], dim=1)  # [1, 32+P, 2048]
    outputs = Qwen2Model.forward(
        backbone,
        input_ids=None,
        inputs_embeds=prefix,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past_kv = outputs.past_key_values
    last_h = outputs.last_hidden_state[:, -1:, :]  # [1, 1, 2048]
    next_logits = lm_head(last_h)[:, 0, :]         # [1, V]

    generated = []
    mtp_drafts = []

    for step in range(max_new_tokens):
        token_id = next_logits.argmax(dim=-1).item()
        generated.append(token_id)
        if token_id == eos_id:
            break

        # Compute MTP drafts from current hidden state
        drafts = []
        h = last_h[:, 0, :]  # [1, 2048]
        z = h
        for k in range(K):
            if k == 0:
                z = mtp_head(h)
            else:
                z = mtp_head(z + h)
            draft_logits = lm_head(z)  # [1, V]
            draft_id = draft_logits.argmax(dim=-1).item()
            drafts.append(draft_id)
        mtp_drafts.append(drafts)

        # Next step: feed token through decoder with KV cache
        token_tensor = torch.tensor([[token_id]], device=device, dtype=torch.long)
        tok_embed = embed_fn(token_tensor)  # [1, 1, 2048]
        outputs = Qwen2Model.forward(
            backbone,
            input_ids=None,
            inputs_embeds=tok_embed,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_kv = outputs.past_key_values
        last_h = outputs.last_hidden_state  # [1, 1, 2048]
        next_logits = lm_head(last_h)[:, 0, :]

    return generated, mtp_drafts


# ---------------------------------------------------------------------------
# MTP acceptance rate
# ---------------------------------------------------------------------------

def compute_acceptance_rates(generated, mtp_drafts, K):
    """
    For each MTP offset k (1-indexed), compute fraction of steps where
    the draft at depth k matches the actual token generated k+1 steps later.

    MTP draft at step t, depth k predicts the token at step t+k+1.
    """
    rates = [0.0] * K
    counts = [0] * K

    for t, drafts in enumerate(mtp_drafts):
        for k in range(min(K, len(drafts))):
            future_idx = t + k + 1
            if future_idx < len(generated):
                counts[k] += 1
                if drafts[k] == generated[future_idx]:
                    rates[k] += 1

    for k in range(K):
        if counts[k] > 0:
            rates[k] /= counts[k]
        else:
            rates[k] = 0.0

    return rates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run MTP checkpoint on held-out set")
    parser.add_argument("--latent_dir", type=str, required=True,
                        help="Directory with .pt shards for the held-out set")
    parser.add_argument("--model_name", type=str,
                        default="liufanfanlff/C3-Context-Cascade-Compression")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Checkpoint directory (contains lora_adapter/ and mtp_head.pt)")
    parser.add_argument("--mtp_k", type=int, default=5)
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument("--prompt", type=str, default="Repeat the text: ")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples (for quick testing)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model, mtp_head, tokenizer = load_model_for_evaluation(
        args.model_name, args.checkpoint_dir, device=args.device)
    eos_id = tokenizer.eos_token_id

    # Tokenize prompt
    prompt_ids = torch.tensor(
        tokenizer.encode(args.prompt), dtype=torch.long, device=device)
    embed_fn = model.base_model.model.model.embed_tokens
    prompt_embeds = embed_fn(prompt_ids).unsqueeze(0)  # [1, P, 2048]

    # Load held-out data
    held_out_data = load_shard_data(args.latent_dir)
    if args.max_examples:
        held_out_data = held_out_data[:args.max_examples]

    # Run
    K = args.mtp_k
    results = []

    print(f"\nRunning on {len(held_out_data)} examples with K={K}...")
    for idx, example in enumerate(held_out_data):
        latents = example["latents"].unsqueeze(0).to(device, dtype=torch.bfloat16)
        target_ids = example["target_ids"].tolist()
        ground_truth_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        input_length = len(target_ids)

        # Generate
        generated_ids, mtp_drafts = generate_with_mtp(
            model, mtp_head, latents, prompt_embeds,
            max_new_tokens=args.max_new_tokens, K=K,
            eos_id=eos_id, device=device,
        )

        # Decode generated text
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Metrics
        exact_match = int(generated_text == ground_truth_text)
        char_dist = char_edit_distance(generated_text, ground_truth_text)
        tok_dist = token_edit_distance(generated_ids, target_ids)
        accept_rates = compute_acceptance_rates(generated_ids, mtp_drafts, K)

        row = {
            "input_length": input_length,
            "exact_match": exact_match,
            "char_edit_dist": char_dist,
            "token_edit_dist": tok_dist,
        }
        for k in range(K):
            row[f"accept_rate_{k+1}"] = round(accept_rates[k], 4)

        results.append(row)

        if (idx + 1) % 50 == 0 or idx == 0:
            em_so_far = sum(r["exact_match"] for r in results) / len(results)
            print(f"  [{idx+1}/{len(held_out_data)}] "
                  f"EM={em_so_far:.3f} | "
                  f"char_dist={char_dist} | tok_dist={tok_dist} | "
                  f"accept_1={accept_rates[0]:.3f}")

    # Write CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = ["input_length", "exact_match", "char_edit_dist", "token_edit_dist"]
    fieldnames += [f"accept_rate_{k+1}" for k in range(K)]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {args.output}")

    # Summary statistics
    n = len(results)
    if n > 0:
        em = sum(r["exact_match"] for r in results) / n
        avg_char = sum(r["char_edit_dist"] for r in results) / n
        avg_tok = sum(r["token_edit_dist"] for r in results) / n
        print(f"\n{'='*60}")
        print(f"Summary ({n} examples):")
        print(f"  Exact match:       {em:.4f}")
        print(f"  Avg char edit dist: {avg_char:.1f}")
        print(f"  Avg tok edit dist:  {avg_tok:.1f}")
        for k in range(K):
            key = f"accept_rate_{k+1}"
            avg_rate = sum(r[key] for r in results) / n
            print(f"  MTP accept rate {k+1}: {avg_rate:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
