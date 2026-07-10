# Corpus format

The PSN (Personal Synaptic Network) build pipeline in `code/psn_build/` and the
prefix-embedding bridge scripts (`code/exp_e_psn_pilot.py`) both expect a corpus
of short first-person "thoughts" as a JSON Lines file: one JSON object per line.

## Schema

Each line is a JSON object with at least a `text` field:

```json
{"text": "A short, self-contained thought, opinion, or observation in your own words."}
```

Optional fields (not required by the code, but useful if you extend it):

- `source` - where the thought came from (a journal entry, a chat transcript, a note).
- `timestamp` - ISO-8601 date/time the thought was recorded.
- `tags` - a list of topic tags.

## What makes a good corpus

- One idea per line. Longer entries get truncated/averaged by the sentence
  encoder (`all-MiniLM-L6-v2`, 384-dim) rather than split, so keep each entry
  to roughly one sentence or a short paragraph.
- First-person, opinionated, or stylistically distinctive text works better
  than neutral reference text - the whole point of the PSN is to capture a
  particular voice or pattern of thought, not general world knowledge.
- Size: the original experiments used a corpus of tens of thousands of
  entries (50K+ neurons). The architecture works at much smaller scale too -
  `code/psn_build/config.py` defaults to 200 neurons, which is enough to
  reproduce the qualitative steering effect on a small sample corpus.

## Bring your own corpus

This repository does **not** include any real personal corpus or any
checkpoint trained on one - see the top-level README for why. `sample_corpus.jsonl`
in this directory is a small set of synthetic, generic example "thoughts" written
for this repo, clearly not derived from any real person, so you can smoke-test the
pipeline before pointing it at your own data.

To replicate the original experiments, assemble your own JSONL file in this
format from your own writing (notes, journal entries, chat history you own,
etc.) and pass its path to the ingestion step described in the main README.
