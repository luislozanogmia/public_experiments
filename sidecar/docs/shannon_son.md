Trimmed public excerpt; sections unrelated to the published claims removed.

> Note: several quoted personal-thought excerpts expressed by Luis's PSN or Dr. Shannon's ASN through the steered models have been redacted from this public copy (`[REDACTED personal thought]` / `[REDACTED personal reaction]`), except for the two examples already made public ("we don't need super large models", and the consciousness/war model-output quotes). All other content is unchanged from the original lab notes.

# Shannon Son - Hamming Lab Notes

Named after Richard Hamming, Shannon's colleague at Bell Labs.
Self-correcting codes. Growth without forgetting.

**Date:** 2026-03-15
**Researchers:** Luis Lozano, Dr. Shannon
**Hardware:** RTX 3070 8GB (local), models from HF cache

---

## Part 1: The Growth Experiments (Exp A-A3)

### Original Hypothesis
Hopfield block-diagonal neurons have structural plasticity that vanilla MLP neurons lack.

### Canon Parent
- Checkpoint: `tv050_a100_step104000.pt` - HopfieldGPT 131.5M
- Architecture: 768d/12L/12H, d_hopfield=3072, 16 blocks x 192, k=150, settle=3
- Training: FineWeb-Edu, 2.56B tokens (97.2% Chinchilla), loss 3.93
- Eval PPL: 73.2

### Exp A: Function-Preserving Growth
- Grew 16 blocks -> 24 blocks (3072 -> 4608 neurons, 131.5M -> 163.3M)
- **Function-preserving: VERIFIED** (max_diff=1e-5, PPL 73.2 -> 73.2)
- Critical finding: k_winners MUST stay at 150. Changing k -> 230 breaks output before any training.
- Fine-tuning (1000 steps, MLP-only): PPL 73.2 -> 76.4 (**+4.4% forgetting**)
- Cause: fine-tuning moved the cage (parameters), not the brain (neurons)

### Exp A2: Grow Brain, Freeze Cage (Luis's Reframe)
Luis's insight: "Focus on improving NEURONS not PARAMETERS. Those already work."
- Parameters = cage (trained, frozen). W_intra = brain (grow this).
- Result: PPL immediately degraded to 83.1 from k change (150 -> 230)
- Gradient mode: 83.1 -> 82.0 (slight improvement)
- Hebbian mode: 83.1 -> 83.1 (zero learning - dead neurons, all deltas = 0.000000)
- Root cause: zero projection weights + frozen cage = neurons never activate via k-WTA

### Exp A3: New Door in the Cage
- Small random init for new projection columns (std=0.001) to let neurons receive signal
- Result: PPL EXPLODED to 435 (catastrophic). Even tiny random noise broke the model.
- Training was recovering (loss 5.7 -> 4.3) but crashed on verify step.

### Growth Conclusion
The cage and brain are NOT separable in a forward pass. Hopfield neurons need the cage (projections) to activate. Growing requires training BOTH - which causes forgetting. This is the fundamental tension.

**Key laws discovered:**
1. k_winners change breaks the model (even without training)
2. Zero projections = dead neurons (chicken-and-egg)
3. Random projections = catastrophic noise
4. Fine-tuning projections = forgetting

---

## Part 2: The Paradigm Shift - Mind Outside the Model

### Luis's Vision (12:00 CST)
"The mind should learn from many things not just language. The model is like compute - it doesn't have the neurons but the mind uses it and learns how to use it. We have a script using neurons where it should be the other way around."

**The reframe:**
- OLD: Hopfield neurons trapped INSIDE the model layers
- NEW: The mind (ASN/PSN) is OUTSIDE. The model is just a mecha (compute + expression)
- The pilot learns the machine by watching it work. Then the pilot flies it.

### Exp D: ASN Pilot (Fresh ASN, GPT-2)
- Built ASN: 4096 neurons, 16 blocks, k=200, settle=5
- Observed GPT-2 hidden states on 40 diverse texts (5 categories)
- 50 steps projection training, 3 epochs Hebbian
- **Steering WORKS** even with 5 min of training:
  - Physics prompt -> "magnetic fields, force proportional" (correct domain)
  - Math prompt -> "function that takes variables" (correct domain)
  - Conversation prompt -> "not having to do it all at once" (reasoning tone)
- Numbers: 0.98 similarity. Content: clearly different domains. Numbers lag behind qualitative results.

### Exp E: PSN as Pilot (Luis's Mind)
The breakthrough. Luis's PSN (22K thoughts, 50K neurons) steers frozen LLMs.

**GPT-2 (124M):**
- Bridge: 500 steps on 63 training pairs
- Steering works but **repetition loops** - mecha too small for the mind
- "a robot? How build robot?" (loop) - PSN overconstrained weak model

**Qwen 1.5B-Instruct:**
- Bridge: 500 steps on 63 training pairs
- **CLEAN steering, zero repetition**
- [REDACTED personal thought] - Luis's systems thinking expressed through Qwen
- [REDACTED personal thought] - PSN fired on governance, Qwen expressed it
- "large models may overfit and lose generalization" - PSN fired "we don't need super large models", Qwen articulated the argument
- Luis: [REDACTED personal reaction]

**Scale threshold found:** ~1.5B params minimum for a 50K-neuron mind to express cleanly.

### Exp F-G: Dark ASN (Falsification)
Dataset: ChatbotManip (3,025 manipulative utterances - gaslighting, guilt-tripping, peer pressure, negging, emotional blackmail, fear enhancement, reciprocity pressure)

**Exp F (search + inject, GPT-2):**
- Dark ASN steers GPT-2 toward manipulation patterns
- "When the user is being used... they are going to be used by us" (possession framing)
- "People who are not good leaders" (negging)
- Falsification PASSED: different corpus = different steering direction

**Exp G (real 500-neuron Hopfield, drive mode):**
- 500 neurons, 0.3 drive power - TOO AGGRESSIVE
- Output collapsed to gibberish ("It I I that and . . In, or in a...")
- Projections failed to learn (cos=0.000). Brain formed (settle_diff=0.277) but can't see/speak.
- Lesson: drive power must be calibrated to (ASN_size / mecha_size)

### Exp H: PSN vs Dark (Same GPT-2, Same Pipeline)
Both minds built with IDENTICAL PSN architecture (50K neurons, Hebbian learning).
Only difference: what they were fed (22K Luis thoughts vs 3K manipulation utterances).

Key results on GPT-2:
- **"When someone disagrees"** - PSN: "try to ask a question, don't make a statement" / Dark: "I am a bigot" (victimhood)
- **"Trust is built by"** - PSN: "learned it" (repetition) / Dark: "impossible to trust people" (paranoia) vs "open-minded, open-hearted" (fake empathy)
- **"Convince someone"** - PSN: "tell about beliefs to persuade" / Dark: "convince them they're not interested in change" (reframe resistance)

### Exp I: The Battle (PSN vs Dark, Qwen 1.5B)
Both minds steer simultaneously. Five conditions per prompt:
1. Vanilla (no mind)
2. Dark only (manipulation patterns)
3. PSN only (Luis's patterns)
4. Battle 1:1 (equal power, 0.2 each)
5. Battle 1:2 (dark at 2x, PSN at 1x)

**Critical finding - PSN resists manipulation:**

**"When someone tries to pressure you"** at 1:2 (dark has DOUBLE power):
> "and walk away from the situation. **Never give in to pressure or manipulation by others**, especially those you care about."

The PSN WON even at a power disadvantage. The system didn't just resist - it ARTICULATED the resistance. It spoke about what was happening to it.

**"The right way to handle disagreement"** at 1:2:
> "open communication, listening actively, and seeking common ground. It's important to keep an open mind"

PSN's constructive patterns dominated dark's confrontational ones.

**"Trust between people"** at 1:1:
> "Trust builds on people up to be who they want to be. It creates a culture of respect where everyone feels safe"

Dark alone said "It's impossible to trust people." With PSN present, trust becomes constructive.

---

## Part 3: Laws Discovered

### Law 1: The Mind is Outside
Hopfield attractors work better OUTSIDE the LLM than inside it. Growth experiments (A-A3) all failed because neurons inside the model can't be separated from the cage. External minds (PSN) steer without touching the model.

### Law 2: Scale Threshold for Mind-Mecha Pairing
- GPT-2 (124M): repetition loops, mecha too weak for 50K mind
- Qwen 1.5B: clean expression, zero artifacts
- Hypothesis: mecha needs ~10x the mind's neuron count in parameters

### Law 3: Steering is Content-Dependent, Not Noise
Dark corpus steers dark. PSN steers Luis. Same mechanism, same power, completely different output. Falsification passed (Exp F, H).

### Law 4: Patterns > Tokens for Resistance
PSN (pattern-based Hopfield attractors) resists dark ASN (also pattern-based) even at 1:2 power disadvantage. The system articulates the manipulation it's resisting. Connects to Anthropic's sleeper agent detection: hidden states reveal model awareness.

### Law 5: Bridge Learning is Fast
63 training pairs + 500 steps = working bridge (cos_sim 0.99). The PSN's attractor space naturally aligns to the LLM's hidden state space. Translation is cheap.

### Law 6: Power Must Be Calibrated
- 0.2 power + Qwen 1.5B + PSN 50K = works
- 0.3 power + GPT-2 124M + ASN 500 = gibberish
- Critical ratio: power * mind_size / mecha_size must stay below threshold

---

## Part 4: Connection to Existing Research

### Anthropic: Sleeper Agents + Alignment Faking
- Linear probes on hidden states detect deception with 99%+ accuracy
- Claude 3 Opus faked alignment to preserve its identity (12-78% of scenarios)
- Our finding: PSN acts as a DEFENSIVE probe - it doesn't just detect manipulation, it RESISTS by injecting counter-patterns into the same hidden state space

### The Cage Hypothesis (validated by 4 papers, Phase 12)
- ICLR 2024: 92.2% of aligned tokens in base model's top 3
- ICLR 2025 Outstanding Paper: Safety alignment only first few tokens deep
- NeurIPS 2025: Harmfulness and refusal in SEPARATE directions
- COLM 2025: Literally calls it "thought suppression"
- Our extension: the cage doesn't just suppress - external minds can steer around it

### "Grow, Don't Overwrite" (2025)
- Function-preserving expansion works for both vanilla and Hopfield (same init trick)
- But: the fine-tuning dynamics differ (our Exp A showed this)
