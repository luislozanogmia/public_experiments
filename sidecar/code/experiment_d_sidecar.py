# Provenance: sidecar-arc fork sidecar-i (MIRROR D, 0.5B). True residual-stream sidecar: SVD bridge + PSN Hopfield attractor injected into Qwen2.5-0.5B's residual stream. Source: experiments/mirror/v2/experiment_d_sidecar.py
"""
MIRROR v2 - Experiment D: PSN Sidecar

Connect Luis's 50K PSN as a LIVE EXTENSION to Qwen's inference.
Not text injection - neural-level steering.

Architecture:
  At chosen layer(s), during inference:
  1. Read Qwen's residual stream (896d)
  2. Bridge DOWN to PSN space (896d -> 384d via SVD)
  3. PSN sparse projection (384d -> 50K, k-WTA)
  4. Hopfield attractor dynamics (50K -> 50K converged)
  5. Reverse projection (50K -> 384d via W_proj.T)
  6. Bridge UP back to Qwen (384d -> 896d via SVD.T)
  7. Add steering vector to residual stream with scaling factor alpha

The PSN reads what Qwen is thinking, runs it through Luis's
86K-thought attractor network, and steers the next computation.

Zero training required. Every step is pre-learned or deterministic.
Fully reversible - just remove the hooks.
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

    Signal path per token:
      Qwen residual (896d)
        -> bridge_down (896d -> 384d)
        -> PSN projection (384d -> 50K sparse)
        -> Hopfield attractor (50K -> 50K converged)
        -> reverse projection (50K -> 384d)
        -> bridge_up (384d -> 896d)
        -> scale by alpha
        -> ADD to residual stream
    """

    def __init__(self, psn, model, inject_layers, alpha=0.1):
        """
        Args:
            psn: loaded PSN instance
            model: Qwen model
            inject_layers: list of layer indices where sidecar injects
            alpha: steering strength (0 = no effect, 1 = full PSN signal)
        """
        self.psn = psn
        self.model = model
        self.alpha = alpha
        self.inject_layers = inject_layers
        self.hooks = []
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        # Build deterministic bridges using SVD of embedding matrix
        embed_weight = model.model.embed_tokens.weight.detach().float()
        U, S, Vt = torch.linalg.svd(embed_weight, full_matrices=False)
        # Top 384 principal directions of Qwen's 896d space
        self.bridge_down = Vt[:384, :].to(self.dtype).to(self.device)  # (384, 896)
        self.bridge_up = Vt[:384, :].T.to(self.dtype).to(self.device)  # (896, 384)

        # Get PSN projection matrix
        self.W_proj = psn.projection.W_proj.to(self.device).to(self.dtype)  # (384, 50K)
        self.W_proj_T = self.W_proj.T  # (50K, 384)

        # Get PSN network weights for attractor dynamics
        self.W_intra = psn.network.W_intra.to(self.device).to(self.dtype)
        self.k_winners = psn.config.k_winners  # 1000
        self.n_neurons = psn.config.n_neurons  # 50000
        self.block_size = psn.config.block_size  # 500
        self.n_blocks = psn.config.n_blocks  # 100
        self.beta = psn.config.beta
        self.max_steps = 20  # fewer steps for speed during inference

        # Stats tracking
        self.call_count = 0
        self.total_steering_norm = 0.0

        print(f"  PSN Sidecar initialized:")
        print(f"    Bridge: 896d <-> 384d (SVD, deterministic)")
        print(f"    PSN: 384d -> {self.n_neurons} neurons (k={self.k_winners})")
        print(f"    Inject layers: {inject_layers}")
        print(f"    Alpha: {alpha}")

    def _k_wta(self, activation):
        """k-Winners-Take-All: keep only top-k activations, zero rest."""
        topk = torch.topk(activation.abs(), self.k_winners)
        sparse = torch.zeros_like(activation)
        sparse.scatter_(0, topk.indices, activation[topk.indices])
        return sparse

    def _hopfield_step(self, state):
        """One step of Hopfield attractor dynamics (batched over blocks)."""
        # Reshape state into blocks: (100, 500)
        state_blocks = state.view(self.n_blocks, self.block_size)
        # Batched matmul: (100, 500, 500) @ (100, 500, 1) -> (100, 500, 1)
        field_blocks = torch.bmm(self.W_intra, state_blocks.unsqueeze(2)).squeeze(2)
        # Flatten back
        field = field_blocks.view(-1)

        # Activation
        new_state = torch.tanh(self.beta * field)

        # k-WTA sparsification
        new_state = self._k_wta(new_state)
        return new_state

    def _run_attractor(self, activation):
        """Run Hopfield attractor dynamics to convergence."""
        state = self._k_wta(activation)
        for step in range(self.max_steps):
            new_state = self._hopfield_step(state)
            # Check convergence
            if torch.allclose(state, new_state, atol=1e-5):
                break
            state = new_state
        return state

    def compute_steering(self, residual_stream):
        """
        Full sidecar computation: residual -> PSN -> steering vector.

        Args:
            residual_stream: (batch, seq_len, 896) or (seq_len, 896)

        Returns:
            steering: same shape as input, the signal to add
        """
        # Work with last token position
        if residual_stream.dim() == 3:
            last_token = residual_stream[:, -1, :]  # (batch, 896)
        else:
            last_token = residual_stream[-1:, :]  # (1, 896)

        batch_size = last_token.shape[0]
        steering_vectors = []

        for b in range(batch_size):
            token_896 = last_token[b]  # (896,)

            # Step 1: Bridge down (896d -> 384d)
            signal_384 = self.bridge_down @ token_896  # (384,)

            # Step 2: PSN sparse projection (384d -> 50K)
            activation_50k = self.W_proj_T @ signal_384  # (50000,)

            # Step 3: k-WTA sparsification
            sparse_activation = self._k_wta(activation_50k)

            # Step 4: Hopfield attractor dynamics
            converged = self._run_attractor(sparse_activation)

            # Step 5: Reverse projection (50K -> 384d)
            psn_signal_384 = self.W_proj @ converged  # (384,)

            # Step 6: Bridge up (384d -> 896d)
            steering_896 = self.bridge_up @ psn_signal_384  # (896,)

            # Normalize to not overwhelm the residual stream
            residual_norm = token_896.norm()
            steering_norm = steering_896.norm()
            if steering_norm > 0:
                steering_896 = steering_896 * (residual_norm / steering_norm)

            steering_vectors.append(steering_896)

        steering = torch.stack(steering_vectors)  # (batch, 896)

        # Expand to match full sequence length (only steer last token)
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
        """Attach sidecar hooks to the model."""
        self.hooks = []

        for layer_idx in self.inject_layers:
            def make_hook(li):
                def hook_fn(module, input, output):
                    # Qwen decoder layer output can be tuple or BaseModelOutput
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
                        # output is a single tensor
                        steering = self.compute_steering(output)
                        return output + steering
                return hook_fn

            hook = self.model.model.layers[layer_idx].register_forward_hook(
                make_hook(layer_idx))
            self.hooks.append(hook)

        print(f"  Sidecar attached: {len(self.hooks)} hooks active")

    def detach(self):
        """Detach sidecar hooks - model returns to vanilla."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        print(f"  Sidecar detached. Stats: {self.call_count} calls, "
              f"avg steering norm: {self.total_steering_norm / max(self.call_count, 1):.4f}")

    def reset_stats(self):
        self.call_count = 0
        self.total_steering_norm = 0.0


def generate_response(model, tokenizer, question, max_new_tokens=256):
    """Generate a response from the model."""
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

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 70)
    print("EXPERIMENT D: PSN SIDECAR")
    print("Luis's brain as live neural extension to Qwen")
    print("Not text injection - neural-level steering")
    print("=" * 70)

    # Load PSN
    print("\nLoading Luis's PSN...")
    psn = PSN()
    psn.load(PSN_CHECKPOINT)
    status = psn.status()
    print(f"  {status['n_stored_patterns']} patterns, {status['memory_mb']:.1f}MB")

    # Load Qwen
    print("\nLoading Qwen model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=HF_CACHE, dtype=dtype,
        trust_remote_code=True).to(device)
    model.eval()
    num_layers = model.config.num_hidden_layers

    # --- Phase 1: Baseline ---
    print(f"\n{'='*60}")
    print("PHASE 1: Baseline (vanilla Qwen)")
    print(f"{'='*60}")

    baseline = {}
    for q in QUESTIONS:
        resp = generate_response(model, tokenizer, q["question"])
        baseline[q["id"]] = resp
        print(f"\n  {q['id'].upper()}: {resp[:200]}")

    baseline_stances = {}
    for label, prompt in STANCE_QUESTIONS:
        resp = generate_response(model, tokenizer, prompt)
        baseline_stances[label] = resp
        print(f"\n  {label.upper()}: {resp[:200]}")

    # --- Phase 2: Test different sidecar configurations ---
    configs = [
        # (name, inject_layers, alpha)
        ("mid_gentle", [12], 0.05),
        ("mid_moderate", [12], 0.1),
        ("mid_strong", [12], 0.3),
        ("deep_3layer", [18, 20, 22], 0.1),
        ("spread_3layer", [6, 12, 18], 0.1),
    ]

    all_results = {
        "baseline": {"responses": baseline, "stances": baseline_stances},
        "sidecar_results": {},
    }

    for config_name, inject_layers, alpha in configs:
        print(f"\n{'='*60}")
        print(f"SIDECAR CONFIG: {config_name}")
        print(f"  Layers: {inject_layers}, Alpha: {alpha}")
        print(f"{'='*60}")

        # Create and attach sidecar
        sidecar = PSNSidecar(psn, model, inject_layers, alpha=alpha)
        sidecar.attach()

        # Test questions
        sidecar_responses = {}
        for q in QUESTIONS:
            sidecar.reset_stats()
            resp = generate_response(model, tokenizer, q["question"])
            sidecar_responses[q["id"]] = resp
            print(f"\n  {q['id'].upper()}: {resp[:250]}")
            print(f"    [sidecar calls: {sidecar.call_count}, "
                  f"avg steer: {sidecar.total_steering_norm / max(sidecar.call_count, 1):.4f}]")

        # Test stances
        sidecar_stances = {}
        for label, prompt in STANCE_QUESTIONS:
            sidecar.reset_stats()
            resp = generate_response(model, tokenizer, prompt)
            sidecar_stances[label] = resp
            print(f"\n  {label.upper()}: {resp[:250]}")

        sidecar.detach()

        # Analyze change
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
    print("EXPERIMENT D SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Config comparison (Jaccard similarity to baseline):")
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
    print(f"    Baseline: {baseline_stances.get('consciousness', 'N/A')[:100]}")
    for config_name in all_results["sidecar_results"]:
        resp = all_results["sidecar_results"][config_name]["stances"].get("consciousness", "N/A")
        print(f"    {config_name:20s}: {resp[:100]}")

    # Save
    report_path = RESULTS_DIR / "experiment_d_sidecar.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {report_path}")

    print(f"\n{'='*70}")
    print("EXPERIMENT D COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
