#!/usr/bin/env python3
# Provenance: Shannon Son fork sidecar-m (Exp E, PSN bridge). Bridges a trained PSN's recalled attractor state into GPT-2/Qwen hidden space via a learned prefix-embedding bridge (prefix-embedding steering, distinct from the residual-stream sidecar above). Requires a PSN checkpoint built with code/psn_build/. Source: experiments/hamming/exp_e_psn_pilot.py
"""
Experiment E: PSN as Pilot.

Luis's PSN (22K thoughts, 50K neurons) already IS a mind.
Don't build a new ASN. Fork the PSN and teach it to fly GPT-2.

Phase A: Extract GPT-2 hidden states for diverse text.
         For each text, also run PSN recall.
         Learn a bridge: PSN_attractor_state -> GPT2_hidden_space.

Phase B: New prompt -> PSN recalls relevant thoughts -> bridge projects
         to GPT-2 space -> inject as steering token -> GPT-2 generates.

The PSN already has the patterns. We just need a translator.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_mecha(name="qwen", device=DEFAULT_DEVICE):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if name == "gpt2":
        model_id = "gpt2"
        d_state = 768
    elif name == "qwen":
        model_id = "Qwen/Qwen2-1.5B-Instruct"
        d_state = 1536
    else:
        raise ValueError(f"Unknown mecha: {name}")

    print(f"Loading mecha ({model_id})...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Mecha loaded and frozen. {n_params/1e6:.0f}M params, d_state={d_state}")
    return model, tokenizer, d_state


def load_psn():
    """Load Luis's PSN. The mind."""
    from psn_build.psn import PSN
    psn = PSN()
    psn.load(Path(os.environ.get("PSN_CHECKPOINT", str(SCRIPT_DIR / "checkpoints" / "psn_latest.pt"))))
    s = psn.status()
    print(f"  PSN loaded: {s['n_neurons']} neurons, {s['n_stored_patterns']} patterns")
    return psn


def psn_recall_state(psn, query, top_k=5):
    """
    Run PSN recall and return the concatenated match info.
    Returns: text of top matches joined together.
    """
    result = psn.recall(query, top_k=top_k)
    texts = []
    for m in result["matches"]:
        t = m["text"].replace("\n", " ").strip()
        if len(t) > 200:
            t = t[:200]
        texts.append(t)
    return texts


def extract_hidden(model, tokenizer, text, layer_idx=6, device=DEFAULT_DEVICE):
    """Extract LLM hidden state for a text."""
    ids = tokenizer.encode(text, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(ids, output_hidden_states=True)
        n_layers = len(outputs.hidden_states)
        # Clamp layer_idx to valid range
        idx = min(layer_idx, n_layers - 1)
        h = outputs.hidden_states[idx].float().mean(dim=1)  # (1, d_state)
    return h


def phase_a_learn_bridge(model, tokenizer, psn, layer_idx=6,
                         bridge_steps=200, lr=1e-3, device=DEFAULT_DEVICE):
    """
    Phase A: Learn the bridge between PSN recall and GPT-2 hidden states.

    For diverse queries:
    1. Run PSN recall -> get thought texts
    2. Run GPT-2 on those texts -> get hidden states
    3. Run GPT-2 on the query -> get query hidden state
    4. Train bridge: mean(thought hidden states) -> query hidden state

    The bridge learns: "when PSN fires these thoughts, GPT-2's brain
    looks like THIS." Translation layer.
    """
    # Diverse queries to train the bridge - rich coverage
    queries = [
        # AI and architecture
        "What is the meaning of artificial intelligence?",
        "How do neural networks learn from data?",
        "Small models can outperform large ones with the right architecture.",
        "Hopfield networks store memories as attractor states in an energy landscape.",
        "The future of AI is not just scale but structure.",
        "Architecture matters more than training tricks at small scale.",
        "The cage hypothesis says parameters suppress what neurons want to express.",
        "Validation before expression ensures the system thinks before it speaks.",
        "A well-scaffolded small model can beat an unguided large one.",
        "Real intelligence requires grounding, not just pattern matching.",
        "The difference between memorization and understanding is generalization.",
        "Transformer attention lets every token talk to every other token.",
        "Language models predict the next token based on context.",
        "Reinforcement learning from human feedback aligns model behavior.",
        "Mixture of experts routes different inputs to specialized subnetworks.",
        # Business and leadership
        "Building a company requires vision and execution.",
        "The most important thing about leadership is making decisions under uncertainty.",
        "A startup needs to find product market fit before scaling.",
        "Good governance requires checks and balances not a single decision maker.",
        "The best teams combine diverse perspectives with shared goals.",
        "Revenue growth without profitability is not sustainable.",
        "Strategic thinking means choosing what NOT to do.",
        "Customer feedback is the most reliable signal for product direction.",
        "Culture is what happens when nobody is watching.",
        "Delegation is not about giving away work but growing capability.",
        # Science and math
        "Mathematics provides the foundation for understanding the universe.",
        "Information theory tells us that surprise carries the most information.",
        "The brain learns by strengthening connections between co-active neurons.",
        "Evolution optimizes through variation and selection over generations.",
        "Entropy measures disorder and the direction of spontaneous processes.",
        "Statistics helps us make decisions under uncertainty with limited data.",
        "The scientific method requires falsifiable hypotheses and controlled experiments.",
        "Calculus describes how quantities change continuously over time.",
        "Probability theory formalizes reasoning about uncertain events.",
        "Physics seeks the simplest laws that explain the most phenomena.",
        # Research methodology
        "The key to research is testing one hypothesis at a time.",
        "Every experiment should have a damage metric alongside the success metric.",
        "Constraint enables creativity rather than limiting it.",
        "The best code is the code that makes itself unnecessary.",
        "Reproducibility is the foundation of trustworthy science.",
        "Ablation studies reveal which components actually matter.",
        "The simplest explanation that fits the data is usually correct.",
        "Scaling laws predict performance as a function of compute and data.",
        # Personal growth and thinking
        "Learning happens at the edge of your comfort zone.",
        "The most important skill is knowing what you don't know.",
        "Writing forces you to think clearly because vague thoughts produce vague sentences.",
        "Reading broadly across disciplines creates unexpected connections.",
        "Persistence matters more than talent for long term success.",
        "Systems thinking sees the whole not just the parts.",
        "First principles reasoning strips away assumptions to find truth.",
        "Teaching something is the best way to deeply understand it.",
        "Feedback loops accelerate learning when the signal is clear and fast.",
        "The gap between knowing and doing is where most people get stuck.",
        # Technology and building
        "Software architecture determines how easy it is to change the system later.",
        "Technical debt accumulates when you optimize for speed over quality.",
        "Open source enables collaboration at scale across organizations.",
        "APIs define the contract between systems that need to work together.",
        "Monitoring and observability are essential for operating reliable systems.",
        "Version control tracks every change and enables collaboration without conflicts.",
        "Testing catches bugs before users do and prevents regressions.",
        "Automation removes human error from repetitive processes.",
        "Documentation is a gift to your future self and your teammates.",
        "The best tools disappear into the workflow and just work.",
    ]

    # Get d_state from model
    sample_h = extract_hidden(model, tokenizer, queries[0], layer_idx, device)
    d_state = sample_h.shape[-1]
    print(f"  d_state={d_state}")

    # Bridge: linear projection in LLM hidden state space
    bridge = nn.Linear(d_state, d_state, bias=True).to(device)
    nn.init.eye_(bridge.weight)
    nn.init.zeros_(bridge.bias)
    optimizer = torch.optim.Adam(bridge.parameters(), lr=lr)

    print(f"\n=== PHASE A: LEARN BRIDGE ===")
    print(f"  Queries: {len(queries)}")
    print(f"  Bridge steps: {bridge_steps}")

    # Pre-compute: for each query, get PSN thoughts and GPT-2 hidden states
    print(f"  Extracting PSN recalls + GPT-2 hidden states...")
    training_pairs = []

    for query in queries:
        # PSN recalls thoughts
        thoughts = psn_recall_state(psn, query, top_k=3)
        if not thoughts:
            continue

        # Get GPT-2 hidden state for each thought
        thought_hiddens = []
        for thought in thoughts:
            h = extract_hidden(model, tokenizer, thought, layer_idx, device)
            thought_hiddens.append(h)

        # Mean of thought hidden states = "what PSN's recall looks like in GPT-2 space"
        psn_in_gpt2 = torch.cat(thought_hiddens, dim=0).mean(dim=0, keepdim=True)  # (1, 768)

        # Target: GPT-2 hidden state for the query itself
        query_hidden = extract_hidden(model, tokenizer, query, layer_idx, device)

        training_pairs.append((psn_in_gpt2, query_hidden))

    print(f"  Training pairs: {len(training_pairs)}")

    # Train the bridge
    print(f"  Training bridge...")
    for step in range(bridge_steps):
        total_loss = 0.0
        for psn_h, query_h in training_pairs:
            optimizer.zero_grad()
            predicted = bridge(psn_h)
            loss = F.mse_loss(predicted, query_h.detach())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if step % 20 == 0:
            avg_loss = total_loss / len(training_pairs)
            # Check cosine similarity
            with torch.no_grad():
                sims = []
                for psn_h, query_h in training_pairs:
                    pred = bridge(psn_h)
                    sim = F.cosine_similarity(pred, query_h).item()
                    sims.append(sim)
                avg_sim = sum(sims) / len(sims)
            print(f"    step={step}/{bridge_steps} | loss={avg_loss:.4f} | cos_sim={avg_sim:.4f}")

    print(f"  Bridge trained.")
    return bridge


def phase_b_steer(model, tokenizer, psn, bridge, layer_idx=6, device=DEFAULT_DEVICE):
    """
    Phase B: PSN steers GPT-2.

    1. New prompt -> PSN recalls relevant thoughts
    2. Encode thoughts via GPT-2 -> get hidden states
    3. Bridge transforms thought-state into steering vector
    4. Inject as prefix token -> GPT-2 generates
    """
    prompts = [
        "The most important thing about building AI systems is",
        "A good leader makes decisions by",
        "The difference between a model that works and one that doesn't is",
        "When I think about the future of technology I believe",
        "The key insight about small models versus large models is",
    ]

    print(f"\n=== PHASE B: PSN STEERS GPT-2 ===")

    results = []
    for prompt in prompts:
        print(f"\n  PROMPT: {prompt}")

        # PSN recalls
        thoughts = psn_recall_state(psn, prompt, top_k=3)
        print(f"  PSN fired:")
        for i, t in enumerate(thoughts):
            print(f"    {i+1}. {t[:120]}")

        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # --- Vanilla ---
        with torch.no_grad():
            vanilla_out = model.generate(
                ids, max_new_tokens=80, temperature=0.7,
                top_k=40, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        vanilla_text = tokenizer.decode(vanilla_out[0][ids.shape[1]:],
                                         skip_special_tokens=True)

        # --- PSN-steered ---
        with torch.no_grad():
            # Encode PSN thoughts in GPT-2 space
            thought_hiddens = []
            for thought in thoughts:
                h = extract_hidden(model, tokenizer, thought, layer_idx, device)
                thought_hiddens.append(h)
            psn_state = torch.cat(thought_hiddens, dim=0).mean(dim=0, keepdim=True)

            # Bridge to steering vector (scaled down to avoid flooding)
            steering = bridge(psn_state) * 0.2  # 20% power

            # Inject as prefix token
            if hasattr(model, 'transformer'):
                input_embeds = model.transformer.wte(ids).float()
            else:
                input_embeds = model.model.embed_tokens(ids).float()
            steering_token = steering.unsqueeze(1)  # (1, 1, d_state)
            steered_embeds = torch.cat([steering_token, input_embeds], dim=1)

            steered_out = model.generate(
                inputs_embeds=steered_embeds.half(),
                max_new_tokens=80, temperature=0.7,
                top_k=40, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        steered_text = tokenizer.decode(steered_out[0][steered_embeds.shape[1]:],
                                         skip_special_tokens=True)

        print(f"  VANILLA: {vanilla_text[:200]}")
        print(f"  PSN-STEERED: {steered_text[:200]}")

        # Similarity
        with torch.no_grad():
            v_ids = tokenizer.encode(vanilla_text[:200], return_tensors="pt").to(device)
            s_ids = tokenizer.encode(steered_text[:200], return_tensors="pt").to(device)
            if v_ids.shape[1] > 2 and s_ids.shape[1] > 2:
                v_h = model(v_ids, output_hidden_states=True).hidden_states[layer_idx].mean(dim=1)
                s_h = model(s_ids, output_hidden_states=True).hidden_states[layer_idx].mean(dim=1)
                sim = F.cosine_similarity(v_h, s_h).item()
                print(f"  SIMILARITY: {sim:.4f}")
            else:
                sim = None

        results.append({
            "prompt": prompt,
            "thoughts": thoughts,
            "vanilla": vanilla_text[:200],
            "steered": steered_text[:200],
            "similarity": sim,
        })

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_idx = 6

    print(f"\n{'='*70}")
    print(f"  EXP E: PSN AS PILOT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Luis's mind flies the mecha.")
    print(f"{'='*70}")

    # Load mecha
    model, tokenizer, d_state = load_mecha("qwen", device)

    # Load mind
    print("\nLoading mind (PSN)...")
    psn = load_psn()

    # Phase A: Learn bridge
    bridge = phase_a_learn_bridge(model, tokenizer, psn,
                                  layer_idx=layer_idx, bridge_steps=500,
                                  lr=1e-3, device=device)

    # Save bridge
    bridge_path = SCRIPT_DIR / "psn_gpt2_bridge.pt"
    torch.save(bridge.state_dict(), bridge_path)
    print(f"  Bridge saved: {bridge_path.name}")

    # Phase B: Steer
    results = phase_b_steer(model, tokenizer, psn, bridge,
                            layer_idx=layer_idx, device=device)

    # Summary
    print(f"\n{'='*70}")
    print(f"  EXP E RESULTS: PSN PILOT")
    print(f"{'='*70}")
    sims = [r['similarity'] for r in results if r['similarity'] is not None]
    if sims:
        avg_sim = sum(sims) / len(sims)
        print(f"  Avg vanilla-steered similarity: {avg_sim:.4f}")

    # Save report
    report = SCRIPT_DIR / "report_exp_e_psn_pilot.txt"
    with open(report, 'w') as f:
        f.write(f"EXP E: PSN AS PILOT\n")
        f.write(f"Date: {datetime.now()}\n\n")
        for r in results:
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(f"PSN THOUGHTS:\n")
            for t in r['thoughts']:
                f.write(f"  - {t[:150]}\n")
            f.write(f"VANILLA: {r['vanilla']}\n")
            f.write(f"PSN-STEERED: {r['steered']}\n")
            f.write(f"SIM: {r['similarity']}\n\n")
    print(f"  Report: {report.name}")


if __name__ == "__main__":
    main()
