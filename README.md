# Discovering Syntactic Failure Cases via Reinforcement Learning

This repository contains the code and result for a COMP 550 NLP final
project on targeted syntactic evaluation of language models.

The project asks whether reinforcement learning can discover harder
subject-verb agreement minimal pairs than a controlled benchmark, while keeping
the generated data distribution-matched to the benchmark. The focus is not on
training a new language model, but on using a lightweight RL search procedure
to find syntactic test cases where existing language models fail.

## Research Question

Can an RL-guided generator find subject-verb agreement examples that are harder
for language models than a controlled benchmark, under matched subtype coverage?

The project tests this by comparing:

- a benchmark controlled agreement dataset;
- a distribution-aware RL-generated agreement dataset;
- the same causal language model scoring protocol across both datasets.

## Project Summary

The work progressed through four main stages:

1. **Controlled benchmark construction**
   - BLiMP-style grammatical/ungrammatical minimal pairs.
   - 240 examples total.
   - Balanced across simple agreement, PP-attractor, and relative-clause cases.

2. **RL-based hard-case generation**
   - A symbolic RL environment searches over agreement templates and subtypes.
   - Rewards prefer model failures, near-boundary decisions, and hard examples.
   - The initial generator collapsed toward a narrow relative-clause region.

3. **Distribution-aware generation**
   - Subtype quotas were added to prevent subtype collapse.
   - The final RL dataset contains 240 examples.
   - The RL dataset matches the benchmark subtype and template distribution.

4. **Benchmark-vs-RL evaluation and sanity checks**
   - The benchmark and RL datasets are evaluated under the same scoring setup.
   - Sanity checks inspect duplicates, lexical collapse, metadata consistency,
     model disagreement, and generated example quality.

## Models Evaluated

The main reported experiments use Hugging Face causal language models:

- `sshleifer/tiny-gpt2`
- `distilgpt2`
- `gpt2`

The scoring protocol compares the normalized log-probability of the grammatical
sentence against the ungrammatical sentence in each minimal pair.

## Metrics

For each model and dataset, the analysis computes:

- **accuracy**: proportion of pairs where the grammatical sentence is preferred;
- **failure rate**: `1 - accuracy`;
- **near-failure rate**: proportion of examples with a small preference margin;
- **mean preference margin**: average grammatical-minus-ungrammatical score.

Subtype analyses additionally group results by:

- `template_type`
- `subtype`
- `dependency_distance_bucket`
- `clause_depth`

## Main Findings

The final matched benchmark-vs-RL comparison shows:

- The RL-generated dataset is harder overall than the benchmark for
  `distilgpt2` and `gpt2`.
- For `gpt2`, accuracy drops from approximately `0.846` on the benchmark to
  `0.792` on the RL dataset.
- For `distilgpt2`, the effect is smaller but directionally consistent:
  accuracy drops from approximately `0.762` to `0.758`.
- `tiny-gpt2` performs extremely poorly on the RL dataset, but sanity analysis
  suggests this is partly a weak-model artifact rather than reliable evidence
  of broad syntactic difficulty.
- The strongest subtype-level difficulty appears in plural-embedded relative
  clauses and selected PP-attractor subtypes.

Important hard subtypes include:

- `object_relative_clause_plural_embedded_that`
- `object_relative_clause_plural_embedded_who`
- `pp_plural_attractor_with`
- `pp_plural_attractor_near`
- `pp_plural_attractor_behind`
- `pp_singular_attractor_with`


## Setup

This project was developed with Python 3.9. Install the package and core
dependencies with:

```powershell
py -m pip install -e .
py -m pip install -r requirements.txt
```

For Hugging Face model evaluation, also install:

```powershell
py -m pip install -r requirements-hf.txt
```

Model files may be downloaded by Hugging Face on first use.

## Reproduce Main Results

Run the distribution-aware RL generation:

```powershell
py -m syntax_rl.rl.distribution_aware_generate --config configs/rl_distribution_aware.yaml
```

Run benchmark-vs-RL comparison:

```powershell
py -m syntax_rl.evaluation.benchmark_vs_rl --config configs/benchmark_vs_rl.yaml
```

Run sanity checks and error analysis:

```powershell
py -m syntax_rl.analysis.sanity_check --config configs/sanity_check.yaml
```

Run multi-model subtype analysis:

```powershell
py -m syntax_rl.analysis.subtype_analysis --config configs/subtype_analysis_multi_model.yaml
```

## Optional Mistral Evaluation

The repository also includes configs for running the open-source
instruction-tuned model `mistralai/Mistral-7B-Instruct-v0.3` through the same
Hugging Face scoring pipeline. This model is much larger than the GPT-2-family
models above, so it may require a GPU or enough CPU RAM plus disk offloading.
It may also download more than 10 GB of model weights on first use.

Start with a small smoke test:

```powershell
py -m syntax_rl.evaluation.benchmark_vs_rl --config configs/benchmark_vs_rl_mistral_smoke.yaml
```

If the smoke test works, run the full 240-vs-240 benchmark-vs-RL comparison:

```powershell
py -m syntax_rl.evaluation.benchmark_vs_rl --config configs/benchmark_vs_rl_mistral_full.yaml
```

For subtype-level Mistral analysis, first try the small matched-subtype run:

```powershell
py -m syntax_rl.evaluation.benchmark_vs_rl --config configs/mistral_subtype_analysis_small_test.yaml
```

Then run the main matched-subtype Mistral analysis:

```powershell
py -m syntax_rl.evaluation.benchmark_vs_rl --config configs/mistral_subtype_analysis.yaml
```

Expected output directories:

- `outputs/benchmark_vs_rl_mistral_smoke/`
- `outputs/benchmark_vs_rl_mistral_full/`
- `outputs/mistral_subtype_analysis_small_test/`
- `outputs/mistral_subtype_analysis/`

