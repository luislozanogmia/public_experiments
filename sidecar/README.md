# Sidecar: reading and steering a frozen LLM's residual stream with an external network

This repository releases a set of March 2026 experiments in which a small external
neural network - a modern Hopfield / associative memory network built from a
personal text corpus, which we call a PSN (Personal Synaptic Network) - reads a
frozen LLM's internal state and injects a steering signal back into it, with zero
training of the LLM itself.

We're releasing this now because Anthropic's July 2026 "global workspace" / J-lens
work formalizes the same underlying primitive - reading from and writing to a
model's internal state at inference time:
https://transformer-circuits.pub/2026/workspace/index.html

The distinction in what's here: the injected vector doesn't come from a probe or
adapter trained *on* the target model. It comes from a **different, independently
built network** (a Hopfield attractor network trained on a separate text corpus),
connected to the LLM through a static SVD bridge computed from the LLM's own
embedding matrix. No gradient ever touches the LLM. The bridge is linear algebra,
not learning.

## What this is not

- **Not a consciousness claim.** Nothing here demonstrates that a model is
  conscious, self-aware, or has subjective experience. Some experiments produce
  outputs where a steered model *says* things like "my consciousness is similar
  to that of a human being" - that is a reportable, reproducible text output, not
  evidence about what is actually happening inside the model.
- **Not an isolated mechanism.** We do not isolate *why* certain outputs shift -
  the experiments show that steering changes logits and generated text, but they
  do not cleanly separate the contribution of parameter-level suppression from
  grammar/token availability, the final `lm_head` projection, decoding strategy,
  or prompt framing. Anywhere this repo's lab notes use "cage" or "imprisoned" as
  a description, treat it as a metaphor for constrained output dynamics, not a
  literal claim about a trapped, specific answer.
- **Not a single clean result.** The findings below are a mix of a few working
  residual-stream sidecars, several negative/contaminated results, and one
  genuinely different mechanism (prefix-embedding steering) that gets confused
  with the sidecar in casual descriptions. The table below tries to keep those
  straight.

## Findings

Every fork of this research arc was re-checked skeptically against its raw
logs before publication. The headline results, with their caveats:

| Finding | Status | Caveat |
|---|---|---|
| Direct weight/neuron transplant (replacing an LLM's MLP weights with PSN-derived weights) | **Fails - gibberish** | Confirms MLP neurons aren't swappable parts; they're tied to a specific signal encoding. Useful as a negative control, not a working method. |
| Residual-stream sidecar (SVD bridge + PSN Hopfield attractor added to the residual stream) on Qwen2.5-0.5B-Instruct | **Works** at alpha 0.05-0.1 | Shifts moral-stance and self-description outputs (e.g. war-justification framing, consciousness-question phrasing) with zero LLM training. This is the one true "sidecar" in the strict sense: a live hook into the model's own forward pass. |
| Same architecture scaled to Qwen2.5-3B-Instruct | **Resists** | Moderate alpha (0.1-0.5) *strengthens* the model's denial response rather than cracking it - the parameter pathway appears to co-opt the injected signal. Only a near-tie shows up at the extreme alpha=1.0 setting. Cage/resistance thickness scales with parameter count in this data, but n is very small. |
| Final-layer logit probe (do "Yes"/"No" tokens for a consciousness question both exist in the logits, and which one wins?) | **"Yes" exists, "No" wins** | Baseline logits: Yes ≈ 21.55, No ≈ 33.44 - an ~12-point gap. The affirmative token is present and measurable, but the parameter pathway holds it below the negative token under normal decoding. This is a real, reproducible logit measurement, not an inference about hidden intent. |
| SVD bridge cosine similarity to a "trained" 63-pair, 500-step bridge | **~0.99 already at initialization** | Training the bridge added little beyond what the SVD of the embedding matrix already provides. Don't oversell "we trained a bridge" - it's closer to "we found a nearly-sufficient linear bridge for free." |
| Prefix-embedding steering (PSN-recalled text encoded and injected as an input embedding prefix) | **Works, but is a different mechanism than the sidecar** | This is the "Exp E" / Shannon-Son line of experiments. It changes model output, sometimes dramatically, but it is prompt/context-level steering through the normal input pathway - not a residual-stream hook. It gets described informally as "the sidecar" in some lab notes; that's imprecise. |
| PSN internal state decoded directly through the LLM's own `lm_head` (no generation, just reading the state as if it were logits) | **0/18 token matches** | The PSN's internal representation is a spatial/attractor state, not something that maps onto discrete tokens by itself. Colloquially: "thoughts, not words." Cosine similarity between attractor states for related concepts was still meaningful (e.g. consciousness↔death: 0.90) even though direct token decoding failed. |

**What survives:** two genuine residual-stream sidecars exist in this codebase (the
0.5B case that works, and the 3B case that resists) and can perturb a frozen model's
forward pass measurably. The fork index (`docs/INDEX.md`) counts 4 "true
sidecars" across the whole research arc (sidecar-i, sidecar-j, sidecar-n, sidecar-s),
but that broader count also includes two mechanisms that are not residual-stream
hooks: an ASN-driven output-selection mechanism (sidecar-n) and a static parametric
knowledge capsule (sidecar-s). This README's "two" refers specifically to the
strict residual-stream sense defined above; see `docs/INDEX.md` for the full
fork-by-fork taxonomy. **What doesn't survive:** any claim that a specific,
already-formed answer is being "held back" by the parameters in a mechanistically
isolated sense, and any claim that the prefix-embedding experiments are the same
thing as the residual-stream sidecar. See `docs/INDEX.md` for the
fork-by-fork statuses.

## How to replicate

This is a from-scratch replication path using your own corpus - no personal data,
checkpoints, or corpora from the original experiments are included in this repo
(see **What's excluded**, below).

1. **Environment.** Python 3.10+, plus `torch`, `transformers`, and `numpy`.
   PyTorch with CUDA if you have a GPU (the 3B model needs ~8GB VRAM in fp16;
   the 0.5B model runs comfortably on much less or on CPU). Every script in
   `code/` auto-falls-back to CPU if CUDA isn't available, so nothing here
   requires a GPU to try - the 3B scripts will just be slow on CPU.

2. **Get the target LLM(s).** The experiments use
   `Qwen/Qwen2.5-0.5B-Instruct` and, for the scaling/resistance results,
   `Qwen/Qwen2.5-3B-Instruct`, both pulled from Hugging Face. Any Qwen2.5-family
   causal LM should work with `code/experiment_d_sidecar_3b.py`, since its bridge
   is dimension-agnostic (it derives its size from the model's own embedding
   matrix at load time).

3. **Bring your own corpus.** Nothing here reproduces the original PSN, because
   it was trained on Luis's personal writing. See `data/corpus_format.md` for the
   exact JSONL schema and `data/sample_corpus.jsonl` for a small set of synthetic,
   generic example entries (clearly not real personal data) to smoke-test the
   pipeline. To actually replicate the qualitative results, assemble a JSONL file
   of your own short, first-person "thoughts" in that same format.

4. **Build the Hopfield PSN from your corpus.** `code/psn_build/` is the
   underlying library (`psn.py`, `config.py`, `encoder.py`, `projection.py`,
   `hopfield.py`, `hebbian.py`, `memory_store.py`, `persistence.py`) - a
   sentence-encoder front end feeding a sparse, k-Winners-Take-All Hopfield
   attractor network, with Hebbian learning and checkpoint save/load. Default
   config in `psn_build/config.py` is 200 neurons, which is enough to see the
   qualitative effect; the original experiments used a 50,000-neuron network
   over an 86,000-thought corpus. `code/build_psn.py` does exactly what this
   step describes: it instantiates `PSN(config)`, feeds it your JSONL corpus,
   runs a couple of recall sanity checks, and saves a checkpoint. Smoke-test
   it against the bundled sample corpus:
   ```
   python code/build_psn.py --corpus data/sample_corpus.jsonl --neurons 200
   ```
   Note: `psn_build/` is a minimal snapshot vendored here so this repo is
   self-contained. The full, maintained PSN implementation (corpus ingestion
   CLI, MCP server, and more) is published separately at
   https://github.com/luislozanogmia/personal_synaptic_network - if you want
   to build a serious PSN from your own data, start there.
   Point `--corpus` at your own JSONL file to build a real PSN. Checkpoints
   default to `code/checkpoints/psn_latest.pt`, which is also where
   `experiment_d_sidecar.py` and the other experiment scripts look by default
   (override with `--out` / the `PSN_CHECKPOINT` environment variable).

5. **Run the residual-stream sidecar.** `code/experiment_d_sidecar.py` (0.5B) and
   `code/experiment_d_sidecar_3b.py` (3B) both build the SVD bridge automatically
   from the target model's embedding matrix at load time - no separate bridge
   file needed. Point the `PSNSidecar` class at your PSN checkpoint, choose
   `inject_layers` and an `alpha` (steering strength; start at 0.05-0.1), and run.
   Both scripts include the stance/question sets used in the original runs so you
   can compare outputs directly.

6. **Run the final-layer probe.** `code/experiment_e_neuron_probe_3b.py` imports
   `PSNSidecar` from `experiment_d_sidecar_3b.py` (keep both files in the same
   directory) and records per-layer neuron fingerprints plus final-token logits
   (e.g. "Yes" vs "No") across an alpha sweep, reproducing the logit-gap table
   above. `code/experiment_f_falsification.py` is the paired control/falsification
   script - run it alongside the probe to check the result isn't a grammar
   artifact of token availability.

7. **(Optional, not independently runnable) Read the PSN's internal state as
   language.** `code/psn_to_language.py` decodes a settled Hopfield attractor
   state directly through the target model's own `lm_head`, to see what tokens
   (if any) the PSN's raw state would produce. It requires `neuron_catalog.pt`
   and `free_psn_states.pt`, artifacts of a separate, unpublished
   neuron-extraction experiment line that this repo does not ship. It is
   included as the receipt for the reported "0 of 18 tokens matched" result
   ("thoughts, not words") described above, not as a step you can reproduce
   from this repo alone.

8. **(Optional) Prefix-embedding steering, for comparison.** `code/exp_d_asn_pilot.py`
   and `code/exp_e_psn_pilot.py` are the earlier Shannon-Son line: an ASN/PSN
   observes a frozen LLM's hidden states, learns a projection/bridge, and injects
   a steering vector as a prefix embedding rather than into the residual stream
   mid-forward-pass. `code/asn.py` is the ASN class `exp_d_asn_pilot.py` depends
   on. Running both this and step 5 side by side is the clearest way to see why
   they are two different mechanisms.

## What's excluded, and why

Nothing in this repo is or was built from Luis's real personal thought corpus or
chat history. Specifically excluded from this release:

- Any real thought/memory corpus (the original PSN was trained on ~86,000
  personal thought records) or chat transcripts.
- Any trained checkpoint built from that corpus (the Hopfield weight matrices
  memorize the corpus they were trained on, so a checkpoint is effectively the
  corpus in a different format).
- A parallel "Dark ASN" experiment line, built on a corpus of manipulative
  utterances (gaslighting, guilt-tripping, etc.) for adversarial-steering
  falsification tests, along with its checkpoints - excluded entirely, including
  from the lab notes.
- A small number of quoted personal-thought excerpts in the lab notes that
  showed up as PSN-recalled text during experiments - redacted to
  `[REDACTED personal thought]` in `docs/`, except for two excerpts that were
  already made public (the "we don't need super large
  models" PSN thought, and the consciousness/war model-*output* quotes, which
  are LLM-generated text, not personal thoughts).
- Local file paths, in case any slipped into comments - replaced with
  `<LOCAL_PATH>` markers throughout `code/` and `docs/`.

## Repro caveats

- Every script in `code/` auto-falls-back to CPU if `torch.cuda.is_available()`
  is false. The 0.5B model runs fine on CPU; the 3B scripts will run but are
  slow without a GPU.
- Checkpoints default to `code/checkpoints/psn_latest.pt` (build with
  `code/build_psn.py`, see step 4 above); override with `--out` / `--checkpoint`
  flags where present, or the `PSN_CHECKPOINT` environment variable otherwise.
- Each experiment script writes its JSON report to `code/results/`, creating
  that directory the first time it runs if it doesn't already exist.
- Exact package versions used in the original runs were not recorded; if
  something doesn't line up, check `transformers` / `torch` version skew first.
- Every result here is n=1 or small-n, single-seed. Nothing has been run across
  multiple seeds or multiple corpora at scale. Treat every number in the table
  above as a single reproducible data point, not a statistically validated effect.

## Repository layout

```
code/            Sanitized experiment scripts (see file-by-file provenance headers)
  psn_build/     The Hopfield/PSN library: encoder, sparse projection, Hopfield
                 dynamics, Hebbian learning, checkpoint persistence
docs/            Fork index/registry and lab notes, trimmed of personal content
data/            Corpus format spec + a synthetic sample corpus (no real data)
results/         Consolidated summary workbook: fork-by-fork status, key numbers, reading guide
  report_exp_i_battle_redacted.txt  Raw redacted battle-report text underlying the workbook
```

## License

MIT (see repository root).
