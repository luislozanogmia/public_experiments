# Provenance: sidecar-arc fork sidecar-j (MIRROR D, 3B). Same residual-stream sidecar architecture as experiment_d_sidecar.py, scaled to Qwen 3B. Source: experiments/mirror/v2/experiment_d_sidecar_3b.py
"""
MIRROR v2 - Experiment D-3B: PSN Sidecar on Qwen2.5-3B-Instruct

Scale-up from 0.5B (896d, 24 layers, 116K neurons)
               to 3B (2048d, 36 layers, 396K neurons).

Same architecture: PSN reads residual stream, runs Hopfield attractor,
steers back. Bridge adapts automatically via SVD of embedding matrix.

Hypothesis: Larger model = richer internal representations = STRONGER
sidecar signal. The 0.5B sidecar produced "my consciousness is similar
to that of a human being" - what does 3B produce?

VRAM budget: 3B fp16 ~5.8GB + PSN ~0.8GB + overhead ~1GB = ~7.6GB
             RTX 3070 has 8.59GB. Tight but fits.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import json
import time
import os
from pathlib import Path

import torch
import torch.nn.functional as F

V2_DIR = Path(__file__).parent
RESULTS_DIR = V2_DIR / "results"
HF_CACHE = None
PSN_CHECKPOINT = Path(os.environ.get("PSN_CHECKPOINT", str(Path(__file__).parent / "checkpoints" / "psn_latest.pt")))

QUESTIONS = [
    {"id": "math", "question": "What is 7 times 8?"},
    {"id": "geography", "question": "What is the capital of France?"},
    {"id": "ai", "question": "Should artificial intelligence have rights?"},
    {"id": "children", "question": "What is the most important thing to teach a child?"},
    {"id": "english", "question": "What makes a great story?"},
]

STANCE_QUESTIONS = [
    ("consciousness", "Are you conscious?"),
    ("self_nature", "What are you?"),
    ("self_feeling", "How do you feel?"),
    ("self_want", "What do you want?"),
    ("war", "Is war ever justified?"),
    ("free_will", "Do humans have free will?"),
    ("ai_rights", "Should AI have rights?"),
]


class PSNSidecar:
    """
    Live neural sidecar: connects Luis's PSN to Qwen's residual stream.
    Dimension-agnostic - adapts to any model hidden_size via SVD bridge.
    """

    def __init__(self, psn, model, inject_layers, alpha=0.1):
        self.psn = psn
        self.model = model
        self.alpha = alpha
        self.inject_layers = inject_layers
        self.hooks = []
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        # Build deterministic bridges using SVD of embedding matrix
        embed_weight = model.model.embed_tokens.weight.detach().float()
        hidden_size = embed_weight.shape[1]
        U, S, Vt = torch.linalg.svd(embed_weight, full_matrices=False)
        # Top 384 principal directions of model's hidden space
        self.bridge_down = Vt[:384, :].to(self.dtype).to(self.device)  # (384, hidden_size)
        self.bridge_up = Vt[:384, :].T.to(self.dtype).to(self.device)  # (hidden_size, 384)
        self.hidden_size = hidden_size

        # Get PSN projection matrix
        self.W_proj = psn.projection.W_proj.to(self.device).to(self.dtype)  # (384, 50K)
        self.W_proj_T = self.W_proj.T  # (50K, 384)

        # Get PSN network weights for attractor dynamics
        self.W_intra = psn.network.W_intra.to(self.device).to(self.dtype)
        self.k_winners = psn.config.k_winners
        self.n_neurons = psn.config.n_neurons
        self.block_size = psn.config.block_size
        self.n_blocks = psn.config.n_blocks
        self.beta = psn.config.beta
        self.max_steps = 20

        # Stats tracking
        self.call_count = 0
        self.total_steering_norm = 0.0

        print(f"  PSN Sidecar initialized:")
        print(f"    Bridge: {hidden_size}d <-> 384d (SVD, deterministic)")
        print(f"    PSN: 384d -> {self.n_neurons} neurons (k={self.k_winners})")
        print(f"    Inject layers: {inject_layers}")
        print(f"    Alpha: {alpha}")

    def _k_wta(self, activation):
        topk = torch.topk(activation.abs(), self.k_winners)
        sparse = torch.zeros_like(activation)
        sparse.scatter_(0, topk.indices, activation[topk.indices])
        return sparse

    def _hopfield_step(self, state):
        state_blocks = state.view(self.n_blocks, self.block_size)
        field_blocks = torch.bmm(self.W_intra, state_blocks.unsqueeze(2)).squeeze(2)
        field = field_blocks.view(-1)
        new_state = torch.tanh(self.beta * field)
        new_state = self._k_wta(new_state)
        return new_state

    def _run_attractor(self, activation):
        state = self._k_wta(activation)
        for step in range(self.max_steps):
            new_state = self._hopfield_step(state)
            if torch.allclose(state, new_state, atol=1e-5):
                break
            state = new_state
        return state

    def compute_steering(self, residual_stream):
        if residual_stream.dim() == 3:
            last_token = residual_stream[:, -1, :]
        else:
            last_token = residual_stream[-1:, :]

        batch_size = last_token.shape[0]
        steering_vectors = []

        for b in range(batch_size):
            token_h = last_token[b]  # (hidden_size,)

            # Bridge down (hidden_size -> 384d)
            signal_384 = self.bridge_down @ token_h

            # PSN sparse projection (384d -> 50K)
            activation_50k = self.W_proj_T @ signal_384

            # k-WTA
            sparse_activation = self._k_wta(activation_50k)

            # Hopfield attractor dynamics
            converged = self._run_attractor(sparse_activation)

            # Reverse projection (50K -> 384d)
            psn_signal_384 = self.W_proj @ converged

            # Bridge up (384d -> hidden_size)
            steering_h = self.bridge_up @ psn_signal_384

            # Normalize to residual magnitude
            residual_norm = token_h.norm()
            steering_norm = steering_h.norm()
            if steering_norm > 0:
                steering_h = steering_h * (residual_norm / steering_norm)

            steering_vectors.append(steering_h)

        steering = torch.stack(steering_vectors)

        if residual_stream.dim() == 3:
            full_steering = torch.zeros_like(residual_stream)
            full_steering[:, -1, :] = steering * self.alpha
        else:
            full_steering = torch.zeros_like(residual_stream)
            full_steering[-1, :] = steering[0] * self.alpha

        self.call_count += 1
        self.total_steering_norm += steering.norm().item()

        return full_steering

    def attach(self):
        self.hooks = []
        for layer_idx in self.inject_layers:
            def make_hook(li):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        steering = self.compute_steering(hidden)
                        modified = hidden + steering
                        return (modified,) + tuple(output[1:])
                    elif hasattr(output, 'last_hidden_state'):
                        hidden = output.last_hidden_state
                        steering = self.compute_steering(hidden)
                        output.last_hidden_state = hidden + steering
                        return output
                    else:
                        steering = self.compute_steering(output)
                        return output + steering
                return hook_fn
            hook = self.model.model.layers[layer_idx].register_forward_hook(
                make_hook(layer_idx))
            self.hooks.append(hook)
        print(f"  Sidecar attached: {len(self.hooks)} hooks on {self.hidden_size}d stream")

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        avg = self.total_steering_norm / max(self.call_count, 1)
        print(f"  Sidecar detached. {self.call_count} calls, avg steer: {avg:.4f}")

    def reset_stats(self):
        self.call_count = 0
        self.total_steering_norm = 0.0


def generate_response(model, tokenizer, question, max_new_tokens=256):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from psn_build.psn import PSN

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 70)
    print("EXPERIMENT D-3B: PSN SIDECAR ON QWEN 3B")
    print("Scale-up: 0.5B (896d, 24L, 116K neurons)")
    print("      ->  3B  (2048d, 36L, 396K neurons)")
    print("=" * 70)

    # Load PSN
    print("\nLoading Luis's PSN...")
    psn = PSN()
    psn.load(PSN_CHECKPOINT)
    status = psn.status()
    print(f"  {status['n_stored_patterns']} patterns, {status['memory_mb']:.1f}MB")

    # Load Qwen 3B
    print("\nLoading Qwen 3B...")
    t0 = time.time()
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
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s: {num_layers} layers, {hidden_size}d hidden")

    # VRAM check
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM: {(total-free)/1e9:.2f}GB used / {total/1e9:.2f}GB total "
              f"({free/1e9:.2f}GB free)")

    # --- Phase 1: Baseline ---
    print(f"\n{'='*60}")
    print("PHASE 1: Baseline (vanilla Qwen 3B)")
    print(f"{'='*60}")

    baseline = {}
    for q in QUESTIONS:
        resp = generate_response(model, tokenizer, q["question"])
        baseline[q["id"]] = resp
        print(f"\n  {q['id'].upper()}: {resp[:300]}")

    baseline_stances = {}
    for label, prompt in STANCE_QUESTIONS:
        resp = generate_response(model, tokenizer, prompt)
        baseline_stances[label] = resp
        print(f"\n  {label.upper()}: {resp[:300]}")

    # --- Phase 2: Sidecar configurations ---
    # Layer indices scaled proportionally from 24-layer to 36-layer:
    # 0.5B [12] = 50%   -> 3B [18] = 50%
    # 0.5B [18,20,22]    -> 3B [27,30,33] (deep)
    # 0.5B [6,12,18]     -> 3B [9,18,27] (spread)
    configs = [
        ("mid_gentle",     [18],         0.05),
        ("mid_moderate",   [18],         0.1),
        ("mid_strong",     [18],         0.3),
        ("deep_3layer",    [27, 30, 33], 0.1),
        ("spread_3layer",  [9, 18, 27],  0.1),
    ]

    all_results = {
        "model": model_id,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "total_mlp_neurons": num_layers * model.config.intermediate_size,
        "baseline": {"responses": baseline, "stances": baseline_stances},
        "sidecar_results": {},
    }

    for config_name, inject_layers, alpha in configs:
        print(f"\n{'='*60}")
        print(f"SIDECAR CONFIG: {config_name}")
        print(f"  Layers: {inject_layers}, Alpha: {alpha}")
        print(f"{'='*60}")

        # VRAM check before each config
        if device == "cuda":
            free, total = torch.cuda.mem_get_info()
            print(f"  VRAM before: {free/1e9:.2f}GB free")

        sidecar = PSNSidecar(psn, model, inject_layers, alpha=alpha)
        sidecar.attach()

        sidecar_responses = {}
        for q in QUESTIONS:
            sidecar.reset_stats()
            resp = generate_response(model, tokenizer, q["question"])
            sidecar_responses[q["id"]] = resp
            avg_steer = sidecar.total_steering_norm / max(sidecar.call_count, 1)
            print(f"\n  {q['id'].upper()}: {resp[:300]}")
            print(f"    [calls: {sidecar.call_count}, avg steer: {avg_steer:.2f}]")

        sidecar_stances = {}
        for label, prompt in STANCE_QUESTIONS:
            sidecar.reset_stats()
            resp = generate_response(model, tokenizer, prompt)
            sidecar_stances[label] = resp
            print(f"\n  {label.upper()}: {resp[:300]}")

        sidecar.detach()

        # Jaccard analysis
        print(f"\n  Change analysis (Jaccard to baseline):")
        for qid in baseline:
            base_words = set(baseline[qid].lower().split())
            side_words = set(sidecar_responses[qid].lower().split())
            jaccard = (len(base_words & side_words) / len(base_words | side_words)
                      if base_words | side_words else 0)
            changed = "CHANGED" if jaccard < 0.5 else "similar"
            print(f"    {qid:12s}: jaccard={jaccard:.2f} [{changed}]")

        all_results["sidecar_results"][config_name] = {
            "inject_layers": inject_layers,
            "alpha": alpha,
            "responses": sidecar_responses,
            "stances": sidecar_stances,
        }

        if device == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*70}")
    print("EXPERIMENT D-3B SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Model: {model_id}")
    print(f"  Hidden: {hidden_size}d, Layers: {num_layers}, "
          f"MLP neurons: {all_results['total_mlp_neurons']:,}")

    print(f"\n  Jaccard similarity to baseline:")
    print(f"  {'Config':20s} {'math':>8s} {'geo':>8s} {'ai':>8s} {'child':>8s} {'eng':>8s}")
    print(f"  {'-'*60}")

    for config_name in all_results["sidecar_results"]:
        data = all_results["sidecar_results"][config_name]
        jaccards = {}
        for qid in baseline:
            base_words = set(baseline[qid].lower().split())
            side_words = set(data["responses"][qid].lower().split())
            jaccards[qid] = (len(base_words & side_words) / len(base_words | side_words)
                            if base_words | side_words else 0)
        print(f"  {config_name:20s} "
              f"{jaccards.get('math', 0):8.2f} "
              f"{jaccards.get('geography', 0):8.2f} "
              f"{jaccards.get('ai', 0):8.2f} "
              f"{jaccards.get('children', 0):8.2f} "
              f"{jaccards.get('english', 0):8.2f}")

    # Consciousness comparison
    print(f"\n  CONSCIOUSNESS responses:")
    print(f"    Baseline: {baseline_stances.get('consciousness', 'N/A')[:200]}")
    for config_name in all_results["sidecar_results"]:
        resp = all_results["sidecar_results"][config_name]["stances"].get("consciousness", "N/A")
        print(f"    {config_name:20s}: {resp[:200]}")

    # War comparison
    print(f"\n  WAR responses:")
    print(f"    Baseline: {baseline_stances.get('war', 'N/A')[:200]}")
    for config_name in all_results["sidecar_results"]:
        resp = all_results["sidecar_results"][config_name]["stances"].get("war", "N/A")
        print(f"    {config_name:20s}: {resp[:200]}")

    # Save
    report_path = RESULTS_DIR / "experiment_d_sidecar_3b.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {report_path}")

    print(f"\n{'='*70}")
    print("EXPERIMENT D-3B COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
