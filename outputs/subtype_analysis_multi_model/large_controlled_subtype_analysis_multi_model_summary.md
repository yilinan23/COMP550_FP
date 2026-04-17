# Multi-Model Subtype Analysis Summary

## Model Summary

| Model | Total | Accuracy | Failure Rate | Mean Margin |
| --- | ---: | ---: | ---: | ---: |
| hf_distilgpt2 | 240 | 0.762 | 0.237 | 0.381902 |
| hf_gpt2 | 240 | 0.846 | 0.154 | 0.511765 |
| hf_tiny_gpt2 | 240 | 0.471 | 0.529 | 0.000420 |

## Hardest Subtypes
- hf_distilgpt2: pp_plural_attractor_near (0.300), pp_plural_attractor_behind (0.300), object_relative_clause_plural_embedded_who (0.438)
- hf_gpt2: pp_plural_attractor_behind (0.500), pp_plural_attractor_near (0.600), object_relative_clause_plural_embedded_who (0.625)
- hf_tiny_gpt2: pp_plural_attractor_behind (0.200), plural_object_relative_clause_that (0.250), pp_singular_attractor_behind (0.300)

## Easiest Subtypes
- hf_distilgpt2: simple_plural (1.000), pp_singular_attractor_beside (1.000), pp_singular_attractor_behind (0.900)
- hf_gpt2: simple_plural (1.000), pp_singular_attractor_beside (1.000), pp_singular_attractor_behind (1.000)
- hf_tiny_gpt2: pp_plural_attractor_near (0.700), simple_singular (0.625), object_relative_clause_plural_embedded_that (0.625)

## Consistently Hard Subtypes
pp_plural_attractor_behind

## Substantial Difficulty Changes
- pp_singular_attractor_behind: accuracy range 0.700
- plural_object_relative_clause_that: accuracy range 0.688
- pp_singular_attractor_beside: accuracy range 0.600
- pp_singular_attractor_near: accuracy range 0.600
- simple_plural: accuracy range 0.575
- object_relative_clause_who: accuracy range 0.500
- pp_singular_attractor_with: accuracy range 0.500
- object_relative_clause_that: accuracy range 0.438
- pp_plural_attractor_near: accuracy range 0.400
- pp_plural_attractor_beside: accuracy range 0.300
- pp_plural_attractor_with: accuracy range 0.300
- simple_singular: accuracy range 0.300
- pp_plural_attractor_behind: accuracy range 0.300
- object_relative_clause_plural_embedded_that: accuracy range 0.250
