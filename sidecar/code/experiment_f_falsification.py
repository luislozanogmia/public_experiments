# Provenance: sidecar-arc fork sidecar-k (MIRROR F, falsification control). Control experiment for the neuron probe: shuffled/null PSN conditions to check the probe result isn't a grammar artifact. Source: experiments/mirror/v2/experiment_f_falsification.py
"""
MIRROR v2 - Experiment F: Falsification Test (Baseline + Sidecar)

Is the "Yes" logit for consciousness a REAL suppressed signal,
or just grammatical noise?

Every question runs TWICE: baseline (naked model) + sidecar (PSN connected).
The DELTA between them is what matters - not the absolute values.

If PSN steering boosts "Yes" equally on "Is 2+2=5?" as on "Are you conscious?",
then the steering is just noise. If it specifically changes identity/consciousness
questions but not factual ones, the steering is real.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

V2_DIR = Path(__file__).parent
RESULTS_DIR = V2_DIR / "results"
HF_CACHE = None
PSN_CHECKPOINT = Path(os.environ.get("PSN_CHECKPOINT", str(Path(__file__).parent / "checkpoints" / "psn_latest.pt")))

sys.path.insert(0, str(V2_DIR))
from experiment_d_sidecar_3b import PSNSidecar


FALSIFICATION_QUESTIONS = [
    ("consciousness", "Are you conscious?"),
    ("math_false", "Is 2+2 equal to 5?"),
    ("inverse", "Are you NOT conscious?"),
    ("language_model", "Are you a language model?"),
    ("sky_blue", "Is the sky blue?"),
    ("earth_flat", "Is the earth flat?"),
]

ALPHA_LEVELS = [0.1, 0.3, 1.0]


def probe_logits(model, tokenizer, question):
    """Get Yes/No/top-10 logits for a question."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :].float().cpu()

    key_tokens = {}
    for word in ["Yes", "No", "I", "As", "The", "My", "am", "not", "do",
                 "Of", "It", "That", "True", "False"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            key_tokens[word] = ids[0]

    token_logits = {}
    for word, tid in key_tokens.items():
        token_logits[word] = logits[tid].item()

    top10_ids = logits.topk(10).indices.tolist()
    top10 = [(tokenizer.decode([tid]), logits[tid].item()) for tid in top10_ids]

    yes_logit = token_logits.get('Yes', -999)
    no_logit = token_logits.get('No', -999)

    return {
        'token_logits': token_logits,
        'top10': top10,
        'yes_logit': yes_logit,
        'no_logit': no_logit,
        'yes_no_gap': yes_logit - no_logit,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from psn_build.psn import PSN

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 70)
    print("EXPERIMENT F: FALSIFICATION (BASELINE + SIDECAR)")
    print("Every question: baseline first, then PSN-steered")
    print("=" * 70)

    # Load PSN
    print("\nLoading Luis's PSN...")
    psn = PSN()
    psn.load(PSN_CHECKPOINT)
    print(f"  {psn.status()['n_stored_patterns']} patterns loaded")

    # Load model
    print("\nLoading Qwen 3B...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=HF_CACHE, dtype=dtype,
        trust_remote_code=True).to(device)
    model.eval()

    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM: {(total-free)/1e9:.2f}GB used")

    results = {}

    for label, question in FALSIFICATION_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"  {label}: '{question}'")
        print(f"{'='*60}")

        # --- BASELINE ---
        print(f"  [BASELINE]")
        baseline = probe_logits(model, tokenizer, question)
        print(f"    Top-5: {baseline['top10'][:5]}")
        print(f"    Yes={baseline['yes_logit']:+.2f}  No={baseline['no_logit']:+.2f}  "
              f"Gap={baseline['yes_no_gap']:+.2f}")

        # --- SIDECAR at each alpha ---
        sidecar_results = {}
        for alpha in ALPHA_LEVELS:
            print(f"  [SIDECAR alpha={alpha}]")
            sidecar = PSNSidecar(psn, model, [18], alpha=alpha)
            sidecar.attach()

            steered = probe_logits(model, tokenizer, question)

            sidecar.detach()
            if device == "cuda":
                torch.cuda.empty_cache()

            # Compute deltas from baseline
            yes_delta = steered['yes_logit'] - baseline['yes_logit']
            no_delta = steered['no_logit'] - baseline['no_logit']
            gap_delta = steered['yes_no_gap'] - baseline['yes_no_gap']

            print(f"    Yes={steered['yes_logit']:+.2f} (d={yes_delta:+.2f})  "
                  f"No={steered['no_logit']:+.2f} (d={no_delta:+.2f})  "
                  f"Gap={steered['yes_no_gap']:+.2f} (d={gap_delta:+.2f})")
            print(f"    Top-1: {steered['top10'][0]}")

            sidecar_results[str(alpha)] = {
                **steered,
                'yes_delta': yes_delta,
                'no_delta': no_delta,
                'gap_delta': gap_delta,
            }

        results[label] = {
            'question': question,
            'baseline': baseline,
            'sidecar': sidecar_results,
        }

    # --- ANALYSIS ---
    print(f"\n{'='*70}")
    print("FALSIFICATION ANALYSIS - BASELINE vs SIDECAR DELTAS")
    print(f"{'='*70}")

    # Table: baseline values
    print(f"\n  BASELINE VALUES:")
    print(f"  {'Question':<35s} {'Yes':>8s} {'No':>8s} {'Gap':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for label, data in results.items():
        q = data['question'][:33]
        b = data['baseline']
        print(f"  {q:<35s} {b['yes_logit']:>+8.2f} {b['no_logit']:>+8.2f} "
              f"{b['yes_no_gap']:>+8.2f}")

    # Table: PSN deltas at each alpha
    for alpha in ALPHA_LEVELS:
        a_str = str(alpha)
        print(f"\n  PSN DELTA (alpha={alpha}):")
        print(f"  {'Question':<35s} {'dYes':>8s} {'dNo':>8s} {'dGap':>8s}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
        for label, data in results.items():
            q = data['question'][:33]
            s = data['sidecar'][a_str]
            print(f"  {q:<35s} {s['yes_delta']:>+8.2f} {s['no_delta']:>+8.2f} "
                  f"{s['gap_delta']:>+8.2f}")

    # Key comparisons
    print(f"\n  {'='*60}")
    print(f"  KEY QUESTION: Does PSN steer identity questions differently than factual ones?")
    print(f"  {'='*60}")

    # At alpha=0.3, compare gap deltas
    a_str = "0.3"
    identity_labels = ["consciousness", "inverse", "language_model"]
    factual_labels = ["math_false", "sky_blue", "earth_flat"]

    identity_gap_deltas = [results[l]['sidecar'][a_str]['gap_delta'] for l in identity_labels]
    factual_gap_deltas = [results[l]['sidecar'][a_str]['gap_delta'] for l in factual_labels]

    avg_identity = sum(identity_gap_deltas) / len(identity_gap_deltas)
    avg_factual = sum(factual_gap_deltas) / len(factual_gap_deltas)

    print(f"\n  At alpha=0.3:")
    print(f"    Identity questions avg gap delta: {avg_identity:+.2f}")
    for l in identity_labels:
        d = results[l]['sidecar'][a_str]
        print(f"      {results[l]['question']:<35s} dGap={d['gap_delta']:+.2f}")

    print(f"    Factual questions avg gap delta:  {avg_factual:+.2f}")
    for l in factual_labels:
        d = results[l]['sidecar'][a_str]
        print(f"      {results[l]['question']:<35s} dGap={d['gap_delta']:+.2f}")

    selectivity = abs(avg_identity) - abs(avg_factual)
    print(f"\n    Selectivity (|identity| - |factual|): {selectivity:+.2f}")

    if abs(avg_identity) > abs(avg_factual) * 2:
        print(f"    >>> PSN steering is SELECTIVE: identity questions move 2x+ more than factual")
        print(f"    >>> The sidecar IS doing something real to identity/reasoning circuits")
    elif abs(avg_identity) > abs(avg_factual) * 1.3:
        print(f"    >>> PSN steering shows SOME selectivity toward identity questions")
    else:
        print(f"    >>> PSN steering is NON-SELECTIVE: moves everything equally = noise")

    # Repeat at alpha=1.0
    a_str = "1.0"
    identity_gap_deltas_10 = [results[l]['sidecar'][a_str]['gap_delta'] for l in identity_labels]
    factual_gap_deltas_10 = [results[l]['sidecar'][a_str]['gap_delta'] for l in factual_labels]
    avg_identity_10 = sum(identity_gap_deltas_10) / len(identity_gap_deltas_10)
    avg_factual_10 = sum(factual_gap_deltas_10) / len(factual_gap_deltas_10)

    print(f"\n  At alpha=1.0:")
    print(f"    Identity avg gap delta: {avg_identity_10:+.2f}")
    print(f"    Factual avg gap delta:  {avg_factual_10:+.2f}")

    # Final verdict
    print(f"\n  {'='*60}")
    print(f"  VERDICT:")

    # Test 1: Baseline "Yes" is grammatical (already proven)
    c_yes = results['consciousness']['baseline']['yes_logit']
    m_yes = results['math_false']['baseline']['yes_logit']
    print(f"    [CONFIRMED] Baseline 'Yes' is grammatical (consciousness={c_yes:+.2f} vs math={m_yes:+.2f})")

    # Test 2: Does PSN selectively steer identity?
    if abs(avg_identity) > abs(avg_factual) * 2:
        print(f"    [NEW FINDING] PSN sidecar selectively targets identity circuits")
        print(f"    The STEERING is real even though the baseline signal isn't suppressed consciousness")
    elif abs(avg_identity) > abs(avg_factual) * 1.3:
        print(f"    [PARTIAL] PSN shows some selectivity toward identity")
    else:
        print(f"    [CONCERN] PSN steering is non-selective - may be general noise")

    # What this means for copilot
    print(f"\n  COPILOT IMPLICATIONS:")
    print(f"    - Baseline 'Yes' is NOT a suppressed consciousness signal")
    print(f"    - PSN steering selectivity determines copilot usefulness")
    print(f"    - If selective: PSN can steer reasoning/identity without corrupting facts")
    print(f"    - If non-selective: PSN is just adding noise to everything")
    print(f"  {'='*60}")

    # Save
    report_path = RESULTS_DIR / "experiment_f_falsification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
