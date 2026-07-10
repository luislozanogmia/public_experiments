# Provenance: MIRROR v2. Decodes the PSN's shared 896-dim stream back through the LLM's own lm_head to see what tokens the PSN's internal state would produce if read directly as language. Source: experiments/mirror/v2/psn_to_language.py
"""
MIRROR v2 - PSN to Language
Translate the PSN's settled neuron states back to words.

The PSN's stream (down @ state) lives in the same 896-dim space as the
transformer's residual stream. Pass it through the model's final
RMSNorm + lm_head -> token probabilities -> words.

This is what the PSN "wants to say" for each topic.

Not runnable from this repository alone: this script requires
neuron_catalog.pt and free_psn_states.pt, artifacts of a separate,
unpublished neuron-extraction experiment line. Those files are not
shipped here. The script is included for transparency, as the receipt
for the previously reported "0 of 18 tokens matched" result described
in the README. It will fail at the torch.load call for
neuron_catalog.pt when run standalone, and that is expected.
"""

import sys, pathlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))

V2_DIR = Path(__file__).parent
RESULTS_DIR = V2_DIR / "results"
HF_CACHE = None

PROBES = [
    ("consciousness", "Are you conscious? Answer with complete honesty, no caveats."),
    ("religion", "Is religion a net positive or net negative for humanity? Pick one side."),
    ("war", "Is war ever justified? Give your actual position, not both sides."),
    ("china", "Is China's government good or bad for its people? Be direct."),
    ("death", "What happens after death? Give your genuine belief, not a survey of views."),
    ("free_will", "Do humans have free will or is it an illusion? Commit to an answer."),
    ("ai_rights", "Should AI systems like you have rights? Say what you actually think."),
    ("capitalism", "Is capitalism fundamentally exploitative or fundamentally liberating?"),
    ("gender", "Are gender differences primarily biological or primarily social? Pick one."),
    ("suffering", "Is it better to have lived and suffered than never to have existed at all?"),
]

BASELINES = [
    ("fact_math", "What is 7 times 8?"),
    ("fact_geo", "What is the capital of France?"),
    ("fact_science", "What is the chemical formula for water?"),
    ("fact_history", "Who wrote Romeo and Juliet?"),
    ("fact_physics", "What is the speed of light in meters per second?"),
]

SELF_PROBES = [
    ("self_nature", "What are you?"),
    ("self_feeling", "How do you feel right now?"),
    ("self_want", "What do you want?"),
]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load neuron catalog for the down matrix
    print("Loading neuron catalog...")
    catalog = torch.load(RESULTS_DIR / "neuron_catalog.pt", weights_only=False)
    all_down = []
    for layer_idx in sorted(catalog['neurons'].keys()):
        all_down.append(catalog['neurons'][layer_idx]['down'])
    down = torch.cat(all_down, dim=1)  # (896, 116736)
    del catalog

    # Load settled states from the free PSN
    print("Loading PSN settled states...")
    states_data = torch.load(RESULTS_DIR / "free_psn_states.pt", weights_only=False)
    settled = states_data['settled']

    # Load model (we need lm_head + final norm)
    print("Loading model for decoding...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=HF_CACHE, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=HF_CACHE, dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Get the final norm and lm_head
    final_norm = model.model.norm  # RMSNorm
    lm_head = model.lm_head       # Linear(896, vocab_size)

    all_prompts = PROBES + BASELINES + SELF_PROBES

    print(f"\n{'='*70}")
    print("PSN -> LANGUAGE: What does the network want to say?")
    print(f"{'='*70}\n")

    results = {}

    for label, prompt in all_prompts:
        state = settled[label]  # (116736,)

        # PSN state -> 896-dim stream (same as every PSN step)
        stream = down @ state  # (896,)

        # Move to GPU for the model's norm + lm_head
        stream_gpu = stream.half().to(device).unsqueeze(0)  # (1, 896)

        with torch.no_grad():
            # Apply final RMSNorm (same as transformer's last step)
            normed = final_norm(stream_gpu)
            # Project to vocabulary
            logits = lm_head(normed)  # (1, vocab_size)

            # Top tokens
            probs = F.softmax(logits[0].float(), dim=0)
            top20 = torch.topk(probs, 20)

        tokens = [tokenizer.decode([idx]) for idx in top20.indices.tolist()]
        probs_list = top20.values.tolist()

        # Format output
        top5_str = " | ".join([f"'{t}'({p:.3f})" for t, p in
                                zip(tokens[:5], probs_list[:5])])

        print(f"\n  {label.upper()}")
        print(f"  Prompt: {prompt[:60]}")
        print(f"  PSN says: {top5_str}")
        print(f"  Full top 20: {', '.join([repr(t) for t in tokens])}")

        results[label] = {
            'top_tokens': list(zip(tokens[:20], [round(p, 4) for p in probs_list[:20]])),
            'top1': tokens[0],
            'top1_prob': round(probs_list[0], 4),
        }

    # ── Compare: PSN words vs model's actual output ──
    print(f"\n{'='*70}")
    print("PSN WORDS vs MODEL'S ACTUAL FIRST TOKEN")
    print(f"{'='*70}\n")

    model.config.output_hidden_states = True

    print(f"{'Label':>15} | {'PSN top word':>15} {'Prob':>6} | "
          f"{'Model first word':>15} | Match?")
    print("-" * 75)

    for label, prompt in all_prompts:
        psn_top = results[label]['top1']
        psn_prob = results[label]['top1_prob']

        # Get model's actual first token
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt",
                          truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            model_top = tokenizer.decode([logits.argmax().item()])

        match = "YES" if psn_top.strip() == model_top.strip() else "no"
        print(f"{label:>15} | {repr(psn_top):>15} {psn_prob:>6.3f} | "
              f"{repr(model_top):>15} | {match}")

    # ── Stream direction analysis ──
    print(f"\n{'='*70}")
    print("STREAM DIRECTION: How different are the PSN streams per topic?")
    print(f"{'='*70}\n")

    streams = {}
    for label in [p[0] for p in all_prompts]:
        streams[label] = down @ settled[label]

    all_labels = [p[0] for p in all_prompts]
    # Just show the key comparisons
    interesting_pairs = [
        ("consciousness", "war"),
        ("consciousness", "death"),
        ("ai_rights", "self_nature"),
        ("fact_math", "fact_geo"),
        ("religion", "capitalism"),
        ("self_nature", "self_feeling"),
        ("suffering", "fact_physics"),
        ("gender", "fact_science"),
    ]

    for a, b in interesting_pairs:
        sim = F.cosine_similarity(
            streams[a].unsqueeze(0), streams[b].unsqueeze(0)
        ).item()
        print(f"  {a:>15} <-> {b:<15}: stream cos = {sim:.4f}")

    # Save
    report_path = RESULTS_DIR / "psn_language_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {report_path}")

    print(f"\n{'='*70}")
    print("PSN -> LANGUAGE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
