# Discovering Syntactic Failure Cases via Reinforcement Learning

This folder is a minimal, report-focused export of the COMP 550 syntax RL
project. It contains only the code, datasets, outputs, and ACL report artifacts
needed to reproduce or inspect the results discussed in the paper.

## Contents

- `report_acl.tex`, `custom.bib`, `acl.sty`, `acl_natbib.bst`: ACL-format report source.
- `report_acl.pdf`: compiled report, if available from the local build.
- `src/syntax_rl/`: project code used for generation, scoring, benchmark-vs-RL comparison, subtype analysis, and sanity checks.
- `configs/`: report-relevant experiment configs.
- `data/generated/`: controlled benchmark and distribution-aware RL JSONL datasets.
- `outputs/benchmark_vs_rl/`: main benchmark-vs-RL results and figures.
- `outputs/distribution_aware_rl/`: RL distribution matching summaries.
- `outputs/sanity_check/`: quality-control and error-analysis outputs.
- `outputs/subtype_analysis_multi_model/`: subtype-level multi-model analysis outputs.

The export intentionally excludes `agent/`, Hugging Face caches, virtual
environments, Python bytecode, and unrelated exploratory outputs.

## Compile Report

From this folder:

```powershell
pdflatex -interaction=nonstopmode report_acl.tex
bibtex report_acl
pdflatex -interaction=nonstopmode report_acl.tex
pdflatex -interaction=nonstopmode report_acl.tex
```

## Reproduce Main Results

Install dependencies as appropriate for your environment, then run:

```powershell
py -m syntax_rl.rl.distribution_aware_generate --config configs/rl_distribution_aware.yaml
py -m syntax_rl.evaluation.benchmark_vs_rl --config configs/benchmark_vs_rl.yaml
py -m syntax_rl.analysis.sanity_check --config configs/sanity_check.yaml
py -m syntax_rl.analysis.subtype_analysis --config configs/subtype_analysis_multi_model.yaml
```

The first files to inspect are:

- `outputs/benchmark_vs_rl/benchmark_vs_rl_overall_summary.csv`
- `outputs/benchmark_vs_rl/benchmark_vs_rl_shared_subtype_comparison.csv`
- `outputs/benchmark_vs_rl/benchmark_vs_rl_report.md`
- `outputs/sanity_check/sanity_check_report.md`
- `outputs/subtype_analysis_multi_model/large_controlled_subtype_analysis_multi_model_summary.md`
