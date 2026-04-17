# Sanity Check Report

## Bottom Line
- Subtype collapse evidence: False (top RL subtype share 0.167)
- Lexical collapse evidence: False (RL top pattern share 0.004)
- Duplicate or near-duplicate evidence: False
- Grammar/metadata issue evidence: False (0 validation errors)
- TinyGPT-2 artifact exploitation evidence: True

## Model Disagreement
- benchmark:all_models_fail: 13
- benchmark:all_models_succeed: 81
- benchmark:mixed_disagreement: 36
- benchmark:only_one_stronger_model_fails: 19
- benchmark:tiny_fails_stronger_succeed: 91
- rl:all_models_fail: 30
- rl:all_models_succeed: 1
- rl:mixed_disagreement: 45
- rl:only_one_stronger_model_fails: 2
- rl:tiny_fails_stronger_succeed: 162

## Stronger-Model Hard Subtypes
- object_relative_clause_plural_embedded_that: {'hf_distilgpt2': 0.6875, 'hf_gpt2': 0.625}
- pp_plural_attractor_with: {'hf_distilgpt2': 0.7, 'hf_gpt2': 0.6}
- object_relative_clause_plural_embedded_who: {'hf_distilgpt2': 0.625, 'hf_gpt2': 0.5}
- pp_plural_attractor_near: {'hf_distilgpt2': 0.4, 'hf_gpt2': 0.5}
- pp_plural_attractor_behind: {'hf_distilgpt2': 0.5, 'hf_gpt2': 0.2}
- object_relative_clause_that: {'hf_distilgpt2': 0.25, 'hf_gpt2': 0.25}
- pp_singular_attractor_with: {'hf_distilgpt2': 0.2, 'hf_gpt2': 0.2}

## Files To Inspect
- sanity_check_pair_validation.csv
- sanity_check_tiny_extreme_failures.csv
- sanity_check_model_disagreement.csv
- sanity_check_manual_tiny_hardest_rl.csv
- sanity_check_manual_gpt2_hardest_rl.csv
