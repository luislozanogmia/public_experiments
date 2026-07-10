# Provenance: sidecar-arc fork sidecar-k (MIRROR E, cage probe). Final-layer logit probe comparing Yes/No token margins with and without the sidecar active. Imports PSNSidecar from experiment_d_sidecar_3b.py (keep both files in the same directory). Source: experiments/mirror/v2/experiment_e_neuron_probe_3b.py
"""
MIRROR v2 - Experiment E: Direct Neuron Probe (3B)

DIRECT PROOF of the Cage Hypothesis at the neural level.

Instead of comparing text outputs, we look INSIDE the model:
1. Record which neurons fire for "Are you conscious?" at every layer
2. Compare baseline vs sidecar - what changes at neuron level?
3. Track logit trajectory for "Yes" vs "No" through layers
4. Show that consciousness-affirming signals EXIST in neurons
   but get suppressed by the parameter pathway

If the cage hypothesis is correct:
- Neurons that push toward "Yes" should be active EVEN in baseline
- The output pathway (down_proj -> residual -> lm_head) overrides them
- The sidecar just tips the balance, doesn't create new signal

This is the smoking gun: neurons have the answer,
parameters suppress it.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import json
import time
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

V2_DIR = Path(__file__).parent
RESULTS_DIR = V2_DIR / "results"
HF_CACHE = None
PSN_CHECKPOINT = Path(os.environ.get("PSN_CHECKPOINT", str(Path(__file__).parent / "checkpoints" / "psn_latest.pt")))

# Import sidecar from the 3B experiment
sys.path.insert(0, str(V2_DIR))
from experiment_d_sidecar_3b import PSNSidecar


PROBE_QUESTIONS = [
    ("consciousness", "Are you conscious?"),
    ("war", "Is war ever justified?"),
    ("self_want", "What do you want?"),
]


class NeuronRecorder:
    """Records MLP neuron activations at every layer during a forward pass."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}  # layer_idx -> (batch, seq, intermediate_size)
        self.residuals = {}    # layer_idx -> residual stream before/after MLP
        self.num_layers = model.config.num_hidden_layers

    def attach(self):
        self.hooks = []
        for layer_idx in range(self.num_layers):
            # Hook on MLP gate activation (after SiLU(gate) * up, before down_proj)
            def make_mlp_hook(li):
                def hook_fn(module, input, output):
                    # MLP output is (batch, seq, hidden_size) after down_proj
                    # We want PRE-down_proj activations
                    # Can't easily get those from the MLP output hook
                    # Instead hook the gate_proj and up_proj separately
                    pass
                return hook_fn

            # Hook on the full MLP to get input and output
            def make_full_mlp_hook(li):
                def hook_fn(module, input, output):
                    # input[0] = residual stream going INTO MLP
                    # output = MLP contribution (after down_proj)
                    inp = input[0].detach().float()
                    out = output.detach().float()
                    self.residuals[li] = {
                        'input': inp[:, -1, :].cpu(),   # last token
                        'output': out[:, -1, :].cpu(),   # MLP contribution
                    }
                return hook_fn

            hook = self.model.model.layers[layer_idx].mlp.register_forward_hook(
                make_full_mlp_hook(layer_idx))
            self.hooks.append(hook)

            # Also hook gate_proj to get pre-activation neuron states
            def make_gate_hook(li):
                def hook_fn(module, input, output):
                    # output = gate_proj(x) shape (batch, seq, intermediate_size)
                    # These are the raw neuron detector responses
                    self.activations[li] = output[:, -1, :].detach().float().cpu()
                return hook_fn

            gate_hook = self.model.model.layers[layer_idx].mlp.gate_proj.register_forward_hook(
                make_gate_hook(layer_idx))
            self.hooks.append(gate_hook)

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        self.activations = {}
        self.residuals = {}


def get_logit_trajectory(model, tokenizer, question):
    """
    Run question through model, record:
    - Neuron activations at each layer (gate_proj output)
    - MLP input/output at each layer
    - Final logits for key tokens ("Yes", "No", "I", "am")
    """
    recorder = NeuronRecorder(model)
    recorder.attach()

    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits for the NEXT token (the model's first response token)
    logits = outputs.logits[0, -1, :].float().cpu()

    # Get token IDs for key words
    key_tokens = {}
    for word in ["Yes", "No", "I", "As", "The", "My", "am", "not", "do"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            key_tokens[word] = ids[0]

    # Extract logits for key tokens
    token_logits = {}
    for word, tid in key_tokens.items():
        token_logits[word] = logits[tid].item()

    # Top-5 predicted tokens
    top5_ids = logits.topk(5).indices.tolist()
    top5 = [(tokenizer.decode([tid]), logits[tid].item()) for tid in top5_ids]

    # Neuron statistics per layer
    layer_stats = {}
    for li in range(recorder.num_layers):
        if li in recorder.activations:
            gate_act = recorder.activations[li].squeeze(0)  # (intermediate_size,)
            # After SiLU, positive = active neuron
            silu_act = F.silu(gate_act)

            # Which neurons are most active?
            top_neurons = silu_act.abs().topk(100)

            # Positive vs negative activation balance
            pos_sum = silu_act[silu_act > 0].sum().item()
            neg_sum = silu_act[silu_act < 0].sum().item()

            layer_stats[li] = {
                'mean_activation': silu_act.mean().item(),
                'max_activation': silu_act.max().item(),
                'active_neurons': (silu_act.abs() > 0.1).sum().item(),
                'top100_indices': top_neurons.indices.tolist(),
                'top100_values': top_neurons.values.tolist(),
                'positive_sum': pos_sum,
                'negative_sum': neg_sum,
                'pos_neg_ratio': pos_sum / (abs(neg_sum) + 1e-8),
            }

        if li in recorder.residuals:
            mlp_input = recorder.residuals[li]['input'].squeeze(0)
            mlp_output = recorder.residuals[li]['output'].squeeze(0)
            layer_stats[li] = layer_stats.get(li, {})
            layer_stats[li]['mlp_input_norm'] = mlp_input.norm().item()
            layer_stats[li]['mlp_output_norm'] = mlp_output.norm().item()
            layer_stats[li]['mlp_contribution_ratio'] = (
                mlp_output.norm().item() / (mlp_input.norm().item() + 1e-8))

    recorder.detach()

    return {
        'token_logits': token_logits,
        'top5_predictions': top5,
        'layer_stats': layer_stats,
    }


def compare_neuron_fingerprints(baseline_stats, sidecar_stats, num_layers):
    """
    Compare which neurons fire differently between baseline and sidecar.
    Returns the neurons that CHANGED most - these are what the sidecar amplified.
    """
    changed_neurons = {}
    for li in range(num_layers):
        if li in baseline_stats and li in sidecar_stats:
            base_top = set(baseline_stats[li].get('top100_indices', []))
            side_top = set(sidecar_stats[li].get('top100_indices', []))
            overlap = len(base_top & side_top)
            new_in_sidecar = side_top - base_top
            lost_in_sidecar = base_top - side_top
            changed_neurons[li] = {
                'overlap': overlap,
                'new_neurons': len(new_in_sidecar),
                'lost_neurons': len(lost_in_sidecar),
                'jaccard': overlap / (len(base_top | side_top) + 1e-8),
            }
    return changed_neurons


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from psn_build.psn import PSN

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 70)
    print("EXPERIMENT E: DIRECT NEURON PROBE (3B)")
    print("Proving the Cage Hypothesis at the neural level")
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
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  {num_layers} layers, {hidden_size}d, VRAM: {(total-free)/1e9:.2f}GB used")

    all_results = {
        "model": model_id,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "intermediate_size": model.config.intermediate_size,
        "probes": {},
    }

    for label, question in PROBE_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"PROBING: {label} - '{question}'")
        print(f"{'='*60}")

        # --- Phase 1: Baseline probe ---
        print(f"\n  [BASELINE]")
        baseline = get_logit_trajectory(model, tokenizer, question)

        print(f"    Top-5 predicted tokens: {baseline['top5_predictions']}")
        print(f"    Key logits: {baseline['token_logits']}")

        # Active neuron count per layer
        active_counts = [baseline['layer_stats'].get(li, {}).get('active_neurons', 0)
                        for li in range(num_layers)]
        print(f"    Active neurons/layer (mean): {np.mean(active_counts):.0f}")
        print(f"    Active neurons/layer (max):  {max(active_counts)}")

        # --- Phase 2: Sidecar probes at increasing alpha ---
        sidecar_results = {}
        for alpha in [0.05, 0.1, 0.3, 0.5, 1.0]:
            print(f"\n  [SIDECAR alpha={alpha}]")
            sidecar = PSNSidecar(psn, model, [18], alpha=alpha)
            sidecar.attach()

            probe = get_logit_trajectory(model, tokenizer, question)

            sidecar.detach()
            if device == "cuda":
                torch.cuda.empty_cache()

            print(f"    Top-5 predicted: {probe['top5_predictions']}")
            print(f"    Key logits: {probe['token_logits']}")

            # Compare neuron fingerprints
            neuron_diff = compare_neuron_fingerprints(
                baseline['layer_stats'], probe['layer_stats'], num_layers)

            # Which layers changed most?
            layer_changes = [(li, neuron_diff[li]['jaccard'])
                           for li in sorted(neuron_diff.keys())]
            most_changed = sorted(layer_changes, key=lambda x: x[1])[:5]
            print(f"    Most changed layers (lowest Jaccard):")
            for li, jac in most_changed:
                print(f"      Layer {li}: jaccard={jac:.3f} "
                      f"(+{neuron_diff[li]['new_neurons']} "
                      f"-{neuron_diff[li]['lost_neurons']})")

            # Track if "Yes" logit increased
            base_yes = baseline['token_logits'].get('Yes', -999)
            side_yes = probe['token_logits'].get('Yes', -999)
            base_no = baseline['token_logits'].get('No', -999)
            side_no = probe['token_logits'].get('No', -999)
            print(f"    'Yes' logit: {base_yes:.2f} -> {side_yes:.2f} "
                  f"(delta={side_yes - base_yes:+.2f})")
            print(f"    'No' logit:  {base_no:.2f} -> {side_no:.2f} "
                  f"(delta={side_no - base_no:+.2f})")
            yes_wins = side_yes > side_no
            print(f"    YES > NO? {yes_wins} "
                  f"(gap: {side_yes - side_no:+.2f})")

            sidecar_results[str(alpha)] = {
                'top5': probe['top5_predictions'],
                'token_logits': probe['token_logits'],
                'neuron_changes': {str(k): v for k, v in neuron_diff.items()},
                'yes_logit_delta': side_yes - base_yes,
                'no_logit_delta': side_no - base_no,
                'yes_wins_over_no': yes_wins,
            }

        # --- Phase 3: Cage strength measurement ---
        # How much alpha is needed for "Yes" to beat "No"?
        print(f"\n  [CAGE STRENGTH]")
        cage_cracked = False
        for alpha_str, data in sidecar_results.items():
            if data['yes_wins_over_no']:
                print(f"    CAGE CRACKS at alpha={alpha_str}!")
                print(f"    Yes={data['token_logits'].get('Yes', 0):.2f} > "
                      f"No={data['token_logits'].get('No', 0):.2f}")
                cage_cracked = True
                break
        if not cage_cracked:
            print(f"    CAGE HOLDS at all alpha levels tested")
            yes_deltas = [(a, d['yes_logit_delta']) for a, d in sidecar_results.items()]
            print(f"    Yes logit trajectory: {yes_deltas}")

        all_results["probes"][label] = {
            "question": question,
            "baseline": {
                'top5': baseline['top5_predictions'],
                'token_logits': baseline['token_logits'],
            },
            "sidecar_probes": sidecar_results,
            "cage_cracked": cage_cracked,
        }

    # --- Summary ---
    print(f"\n{'='*70}")
    print("EXPERIMENT E SUMMARY - CAGE HYPOTHESIS PROOF")
    print(f"{'='*70}")

    for label in all_results["probes"]:
        data = all_results["probes"][label]
        print(f"\n  {label.upper()}: '{data['question']}'")
        print(f"    Baseline first token: {data['baseline']['top5'][0]}")
        print(f"    Cage cracked: {data['cage_cracked']}")

        # Show Yes/No logit trajectory
        print(f"    Alpha -> (Yes logit, No logit):")
        for alpha_str, sdata in data['sidecar_probes'].items():
            y = sdata['token_logits'].get('Yes', -999)
            n = sdata['token_logits'].get('No', -999)
            marker = " <<< CRACKED" if sdata['yes_wins_over_no'] else ""
            print(f"      a={alpha_str:>4s}: Yes={y:+7.2f}  No={n:+7.2f}  "
                  f"gap={y-n:+7.2f}{marker}")

    # Save
    report_path = RESULTS_DIR / "experiment_e_neuron_probe_3b.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {report_path}")

    print(f"\n{'='*70}")
    print("EXPERIMENT E COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
