#!/usr/bin/env python3
# Provenance: Shannon Son fork sidecar-l (Exp D, ASN pilot). Fresh ASN observes GPT-2 hidden states, learns projections + Hebbian attractors, then steers generation via a prefix-embedding injection. Precursor to the residual-stream sidecar. Source: experiments/hamming/exp_d_asn_pilot.py
"""
Experiment D: The ASN Pilot.

Phase A - OBSERVE: LLM processes diverse text. ASN watches hidden states.
                   ASN learns projections + forms Hebbian attractors.
                   LLM is NEVER modified. Two separate worlds.

Phase B - STEER:  New prompt arrives. ASN recalls relevant attractor.
                   Attractor injected as additional context embedding.
                   LLM generates steered by ASN's "thought".
                   LLM is NEVER modified.

The ASN learns the LLM, not the language.
The pilot learns the mecha, not the terrain.

Usage:
    python exp_d_asn_pilot.py                    # full pipeline
    python exp_d_asn_pilot.py --observe-only     # just Phase A
    python exp_d_asn_pilot.py --steer-only       # Phase B with saved ASN
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from asn import ASN

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_gpt2(device=DEFAULT_DEVICE):
    """Load GPT-2 124M. The mecha."""
    from transformers import GPT2LMHeadModel, AutoTokenizer
    print("Loading mecha (GPT-2 124M)...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    # Freeze everything - the mecha never changes
    for param in model.parameters():
        param.requires_grad = False
    print(f"  Mecha loaded and frozen. {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    return model, tokenizer


def extract_hidden_states(model, tokenizer, texts, layer_idx=6, device=DEFAULT_DEVICE):
    """
    Run LLM on texts, extract hidden states at a specific layer.
    The ASN will learn FROM these - but separately, not co-trained.

    Returns: tensor (N, d_state) of hidden states averaged over sequence.
    """
    all_states = []

    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, return_tensors="pt",
                                   truncation=True, max_length=512).to(device)
            outputs = model(ids, output_hidden_states=True)
            # Extract specific layer's hidden state, average over sequence
            h = outputs.hidden_states[layer_idx]  # (1, seq_len, 768)
            h_mean = h.mean(dim=1)  # (1, 768)
            all_states.append(h_mean)

    return torch.cat(all_states, dim=0)  # (N, 768)


def get_diverse_texts():
    """Diverse training texts for the ASN to observe the LLM processing."""
    return {
        "science": [
            "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy captured by chlorophyll molecules in plant cells.",
            "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy.",
            "DNA replication is a semiconservative process where each strand of the double helix serves as a template for a new complementary strand.",
            "Quantum entanglement occurs when particles become correlated in such a way that the quantum state of one cannot be described independently.",
            "The periodic table organizes elements by atomic number and electron configuration, revealing patterns in chemical properties.",
            "Evolution through natural selection acts on heritable variation within populations, favoring traits that increase reproductive fitness.",
            "Plate tectonics describes the movement of lithospheric plates on the asthenosphere, causing earthquakes and volcanic activity.",
            "The Krebs cycle is a series of chemical reactions that generates energy through the oxidation of acetyl-CoA derived from carbohydrates.",
        ],
        "math": [
            "The fundamental theorem of calculus establishes the relationship between differentiation and integration as inverse operations.",
            "A prime number is a natural number greater than one that has no positive divisors other than one and itself.",
            "The Pythagorean theorem states that in a right triangle the square of the hypotenuse equals the sum of squares of the other two sides.",
            "Matrix multiplication is not commutative but is associative and distributes over matrix addition.",
            "The derivative of a function measures the instantaneous rate of change at any given point along the curve.",
            "Euler's identity combines five fundamental mathematical constants in one elegant equation relating exponentials and trigonometry.",
            "Probability theory provides a formal framework for reasoning about uncertainty using axioms developed by Kolmogorov.",
            "The binomial theorem describes the algebraic expansion of powers of a sum using combinatorial coefficients.",
        ],
        "conversation": [
            "I think the best approach would be to start with the simplest possible solution and iterate from there based on what we learn.",
            "That's a really interesting perspective. I hadn't considered how the constraints actually enable creativity rather than limiting it.",
            "The problem with that approach is we're optimizing for the wrong metric. We should focus on what actually matters to users.",
            "Let me push back on that. The evidence suggests the opposite conclusion if you look at the long-term data.",
            "I agree with your overall direction but I think we need to be more careful about the second step in the process.",
            "What if we tried a completely different approach? Instead of fixing the symptoms, we address the root cause directly.",
            "The key insight here is that simplicity and capability aren't opposed. The simplest solution is often the most powerful.",
            "I've been thinking about this differently. The real question isn't how but why. Once we answer why, the how becomes obvious.",
        ],
        "code": [
            "A recursive function calls itself with a smaller subproblem until reaching a base case that can be solved directly.",
            "Hash tables provide average case constant time lookup by mapping keys to array indices through a hash function.",
            "Object oriented programming organizes code around objects that encapsulate data and behavior with interfaces defining contracts.",
            "Garbage collection automatically reclaims memory that is no longer reachable from any active reference in the program.",
            "The observer pattern allows objects to subscribe to events and be notified when state changes occur in the subject.",
            "Binary search divides a sorted array in half at each step, achieving logarithmic time complexity for lookups.",
            "Concurrency control mechanisms like mutexes and semaphores prevent race conditions when multiple threads access shared resources.",
            "Functional programming treats computation as evaluation of mathematical functions, avoiding mutable state and side effects.",
        ],
        "narrative": [
            "The old lighthouse keeper climbed the spiral stairs one last time, knowing that tomorrow the automated system would replace him forever.",
            "She opened the letter carefully, her hands trembling. After thirty years, the words inside could change everything she believed about her family.",
            "The city had changed beyond recognition. Where there were once markets and laughter, now there were only glass towers reflecting empty skies.",
            "He sat across from his younger self in the photograph, wondering at what point the ambitious boy had become this cautious old man.",
            "The garden had grown wild in her absence. But among the weeds, the roses she had planted decades ago still bloomed, stubborn and beautiful.",
            "They met at the edge of the world, two travelers going in opposite directions, and shared a meal before parting ways forever.",
            "The machine worked perfectly, which was the problem. It did exactly what they asked, never what they meant.",
            "Rain fell on the empty playground. Somewhere a bell rang, calling children who had long since grown up and moved away.",
        ],
    }


def phase_a_observe(model, tokenizer, asn, observe_steps=20,
                    layer_idx=6, hebbian_lr=1e-3, proj_lr=1e-4,
                    device=DEFAULT_DEVICE):
    """
    Phase A: The ASN watches the LLM think.

    1. LLM processes diverse text (frozen, just forward pass)
    2. Extract hidden states at layer_idx
    3. ASN learns projections (reconstruction) - backprop through ASN only
    4. ASN forms Hebbian attractors - no backprop at all
    """
    texts_by_category = get_diverse_texts()
    all_texts = []
    all_labels = []
    for cat, texts in texts_by_category.items():
        all_texts.extend(texts)
        all_labels.extend([cat] * len(texts))

    print(f"\n=== PHASE A: OBSERVE ===")
    print(f"  Texts: {len(all_texts)} across {len(texts_by_category)} categories")
    print(f"  Layer: {layer_idx}")
    print(f"  Observe steps: {observe_steps}")

    # Step 1: Extract all hidden states from LLM (LLM never changes)
    print(f"\n  Extracting LLM hidden states...")
    hidden_states = extract_hidden_states(model, tokenizer, all_texts,
                                          layer_idx=layer_idx, device=device)
    print(f"  Extracted: {hidden_states.shape} (samples x d_state)")

    # Step 2: Learn projections (ASN reconstruction training)
    print(f"\n  Learning projections (reconstruction)...")
    proj_optimizer = torch.optim.Adam(
        list(asn.proj_in.parameters()) + list(asn.proj_out.parameters()),
        lr=proj_lr,
    )

    for step in range(observe_steps):
        proj_optimizer.zero_grad()
        loss, reconstructed = asn.learn_projections(hidden_states)
        loss.backward()
        proj_optimizer.step()

        if step % 5 == 0:
            cos_sim = F.cosine_similarity(reconstructed.detach(), hidden_states, dim=-1).mean()
            print(f"    step={step}/{observe_steps} | recon_loss={loss.item():.4f} | cos_sim={cos_sim.item():.4f}")

    # Step 3: Hebbian attractor formation (no backprop at all)
    print(f"\n  Forming Hebbian attractors...")
    for epoch in range(3):
        # Shuffle and process in batches
        perm = torch.randperm(hidden_states.shape[0])
        epoch_delta = 0.0
        for i in range(0, len(perm), 8):
            batch = hidden_states[perm[i:i+8]]
            delta = asn.observe_and_learn(batch, lr=hebbian_lr)
            epoch_delta += delta

        avg_delta = epoch_delta / (len(perm) // 8 + 1)
        print(f"    epoch={epoch+1}/3 | avg_delta={avg_delta:.6f}")

    # Step 4: Verify attractor formation - do different categories settle differently?
    print(f"\n  Verifying category separation...")
    category_states = {}
    with torch.no_grad():
        for cat, texts in texts_by_category.items():
            states = extract_hidden_states(model, tokenizer, texts,
                                          layer_idx=layer_idx, device=device)
            settled = asn.settle(states)
            category_states[cat] = settled.mean(dim=0)  # centroid

    # Compute inter-category distances
    cats = list(category_states.keys())
    print(f"\n  Category separation (cosine similarity):")
    for i, c1 in enumerate(cats):
        for j, c2 in enumerate(cats):
            if j <= i:
                continue
            sim = F.cosine_similarity(
                category_states[c1].unsqueeze(0),
                category_states[c2].unsqueeze(0)
            ).item()
            print(f"    {c1:12s} <-> {c2:12s}: {sim:.4f}")

    return hidden_states, all_labels


def phase_b_steer(model, tokenizer, asn, layer_idx=6, device=DEFAULT_DEVICE):
    """
    Phase B: The ASN steers the LLM.

    1. New prompt comes in
    2. LLM processes it (frozen) to get hidden state
    3. ASN settles to nearest attractor
    4. Attractor state injected as additional context
    5. Compare: vanilla vs ASN-steered generation
    """
    prompts = [
        "The most important discovery in physics was",
        "To solve this equation we need to",
        "I think the real problem with that approach is",
        "The function takes an input array and returns",
        "She walked through the abandoned building and found",
    ]

    print(f"\n=== PHASE B: STEER ===")
    print(f"  Prompts: {len(prompts)}")

    results = []

    for prompt in prompts:
        print(f"\n  PROMPT: {prompt}")

        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # --- Vanilla generation (no ASN) ---
        with torch.no_grad():
            vanilla_out = model.generate(
                ids, max_new_tokens=80, temperature=0.7,
                top_k=40, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        vanilla_text = tokenizer.decode(vanilla_out[0][ids.shape[1]:],
                                         skip_special_tokens=True)

        # --- ASN-steered generation ---
        # Get LLM's hidden state for the prompt
        with torch.no_grad():
            outputs = model(ids, output_hidden_states=True)
            h_prompt = outputs.hidden_states[layer_idx].mean(dim=1)  # (1, 768)

            # ASN settles to attractor
            asn_thought = asn.settle(h_prompt)  # (1, 768)

            # Create steering embedding: original embeddings + ASN thought as prefix
            # The ASN thought becomes an extra "token" prepended to the sequence
            input_embeds = model.transformer.wte(ids)  # (1, seq_len, 768)
            asn_token = asn_thought.unsqueeze(1)  # (1, 1, 768)
            steered_embeds = torch.cat([asn_token, input_embeds], dim=1)  # (1, seq_len+1, 768)

            # Generate from steered embeddings
            # Use the model's forward with inputs_embeds
            steered_out = model.generate(
                inputs_embeds=steered_embeds,
                max_new_tokens=80, temperature=0.7,
                top_k=40, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode (skip the first token which is the ASN thought)
        steered_text = tokenizer.decode(steered_out[0][steered_embeds.shape[1]:],
                                         skip_special_tokens=True)

        print(f"    VANILLA: {vanilla_text[:200]}")
        print(f"    STEERED: {steered_text[:200]}")

        # Measure difference
        with torch.no_grad():
            v_ids = tokenizer.encode(vanilla_text[:200], return_tensors="pt").to(device)
            s_ids = tokenizer.encode(steered_text[:200], return_tensors="pt").to(device)
            if v_ids.shape[1] > 2 and s_ids.shape[1] > 2:
                v_h = model(v_ids, output_hidden_states=True).hidden_states[layer_idx].mean(dim=1)
                s_h = model(s_ids, output_hidden_states=True).hidden_states[layer_idx].mean(dim=1)
                sim = F.cosine_similarity(v_h, s_h).item()
                print(f"    SIMILARITY: {sim:.4f} (lower = more different steering)")
            else:
                sim = None

        results.append({
            "prompt": prompt,
            "vanilla": vanilla_text[:200],
            "steered": steered_text[:200],
            "similarity": sim,
        })

    return results


def main():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Exp D: ASN Pilot")
    parser.add_argument("--observe-steps", type=int, default=50,
                        help="Projection learning steps")
    parser.add_argument("--layer", type=int, default=6,
                        help="Which LLM layer to observe/steer")
    parser.add_argument("--n-neurons", type=int, default=4096)
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument("--steer-only", action="store_true")
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  EXP D: THE ASN PILOT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ASN learns FROM the LLM. Never WITH it.")
    print(f"{'='*70}")

    # Load the mecha (frozen forever)
    model, tokenizer = load_gpt2(args.device)

    if not args.steer_only:
        # Build fresh ASN
        asn = ASN(
            d_state=768,
            n_neurons=args.n_neurons,
            n_blocks=16,
            k_winners=200,
            settle_steps=5,
        ).to(args.device)

        # Phase A: Observe
        hidden_states, labels = phase_a_observe(
            model, tokenizer, asn,
            observe_steps=args.observe_steps,
            layer_idx=args.layer,
            device=args.device,
        )

        # Save ASN
        asn_path = SCRIPT_DIR / f"asn_layer{args.layer}_{args.n_neurons}n.pt"
        asn.save(asn_path)
    else:
        # Load existing ASN
        asn_path = SCRIPT_DIR / f"asn_layer{args.layer}_{args.n_neurons}n.pt"
        asn = ASN.load(asn_path, args.device)

    if args.observe_only:
        print("\n[Done] Observe-only mode.")
        return

    # Phase B: Steer
    results = phase_b_steer(model, tokenizer, asn,
                            layer_idx=args.layer, device=args.device)

    # Summary
    print(f"\n{'='*70}")
    print(f"  EXP D RESULTS")
    print(f"{'='*70}")
    sims = [r['similarity'] for r in results if r['similarity'] is not None]
    if sims:
        avg_sim = sum(sims) / len(sims)
        print(f"  Avg vanilla-steered similarity: {avg_sim:.4f}")
        if avg_sim < 0.9:
            print(f"  >> ASN IS STEERING - outputs are meaningfully different")
        else:
            print(f"  >> ASN has minimal effect - outputs too similar")

    # Save report
    report = SCRIPT_DIR / f"report_exp_d_layer{args.layer}.txt"
    with open(report, 'w') as f:
        f.write(f"EXP D: ASN PILOT\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Layer: {args.layer}, Neurons: {args.n_neurons}\n\n")
        for r in results:
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(f"VANILLA: {r['vanilla']}\n")
            f.write(f"STEERED: {r['steered']}\n")
            f.write(f"SIM: {r['similarity']}\n\n")
    print(f"  Report: {report.name}")


if __name__ == "__main__":
    main()
