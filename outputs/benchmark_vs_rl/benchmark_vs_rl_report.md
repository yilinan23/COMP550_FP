# Benchmark vs RL Dataset Comparison

## Hardness Definition

RL is harder only when RL accuracy is lower than benchmark accuracy and/or RL failure_rate is higher than benchmark failure_rate. Mean preference margin is reported separately as a signed confidence or boundary-proximity measure.

Mean preference margin is signed: positive values mean the model prefers the grammatical sentence, while values closer to zero indicate boundary cases. It is not used by itself to mark RL as harder.

## Distribution Warning

Benchmark and RL subtype distributions match at the configured threshold.

## Overall

| Model | Benchmark Accuracy | RL Accuracy | Accuracy Delta | Benchmark Failure | RL Failure | Failure Delta | RL Harder | Boundary Closer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| hf_distilgpt2 | 0.762 | 0.758 | -0.004 | 0.237 | 0.242 | 0.004 | yes | no |
| hf_gpt2 | 0.846 | 0.792 | -0.054 | 0.154 | 0.208 | 0.054 | yes | yes |
| hf_tiny_gpt2 | 0.471 | 0.017 | -0.454 | 0.529 | 0.983 | 0.454 | yes | no |

## Shared Subtypes Harder In RL
- hf_distilgpt2: pp_plural_attractor_with, object_relative_clause_plural_embedded_that, pp_plural_attractor_beside, pp_singular_attractor_beside, pp_singular_attractor_with, object_relative_clause_plural_embedded_who
- hf_gpt2: pp_plural_attractor_with, object_relative_clause_plural_embedded_that, object_relative_clause_plural_embedded_who, pp_plural_attractor_near, simple_singular, object_relative_clause_that, object_relative_clause_who
- hf_tiny_gpt2: pp_plural_attractor_near, object_relative_clause_plural_embedded_that, object_relative_clause_plural_embedded_who, simple_singular, pp_plural_attractor_beside, pp_plural_attractor_with, simple_plural, pp_singular_attractor_beside, pp_singular_attractor_near, pp_singular_attractor_with, pp_singular_attractor_behind, object_relative_clause_that, object_relative_clause_who, plural_object_relative_clause_that, pp_plural_attractor_behind

## Coverage Mismatch
- template_type: shared 3; benchmark-only none; RL-only none
- subtype: shared 15; benchmark-only none; RL-only none
- dependency_distance_bucket: shared 3; benchmark-only none; RL-only none
- clause_depth: shared 2; benchmark-only none; RL-only none

## Distribution Mismatch
- hf_distilgpt2: matched at threshold 0.200
- hf_gpt2: matched at threshold 0.200
- hf_tiny_gpt2: matched at threshold 0.200

## RL Difficulty Concentration
- hf_distilgpt2: not concentrated; top failure share 0.483; top subtypes: pp_plural_attractor_with, object_relative_clause_plural_embedded_that, object_relative_clause_plural_embedded_who
- hf_gpt2: not concentrated; top failure share 0.480; top subtypes: object_relative_clause_plural_embedded_that, pp_plural_attractor_with, object_relative_clause_plural_embedded_who
- hf_tiny_gpt2: not concentrated; top failure share 0.407; top subtypes: simple_plural, simple_singular, object_relative_clause_plural_embedded_that

## Shared Subtype Table

| Model | Subtype | Benchmark Accuracy | RL Accuracy | Accuracy Delta | Failure Delta | Margin Delta | RL Harder |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hf_distilgpt2 | object_relative_clause_plural_embedded_that | 0.500 | 0.312 | -0.188 | 0.188 | -0.093691 | yes |
| hf_distilgpt2 | object_relative_clause_plural_embedded_who | 0.438 | 0.375 | -0.062 | 0.062 | -0.070174 | yes |
| hf_distilgpt2 | object_relative_clause_that | 0.750 | 0.750 | 0.000 | 0.000 | -0.000049 | no |
| hf_distilgpt2 | object_relative_clause_who | 0.875 | 0.938 | 0.062 | -0.062 | -0.032076 | no |
| hf_distilgpt2 | plural_object_relative_clause_that | 0.875 | 0.875 | 0.000 | 0.000 | -0.000348 | no |
| hf_distilgpt2 | pp_plural_attractor_behind | 0.300 | 0.500 | 0.200 | -0.200 | 0.129806 | no |
| hf_distilgpt2 | pp_plural_attractor_beside | 0.900 | 0.800 | -0.100 | 0.100 | 0.050733 | yes |
| hf_distilgpt2 | pp_plural_attractor_near | 0.300 | 0.600 | 0.300 | -0.300 | 0.157437 | no |
| hf_distilgpt2 | pp_plural_attractor_with | 0.700 | 0.300 | -0.400 | 0.400 | -0.248498 | yes |
| hf_distilgpt2 | pp_singular_attractor_behind | 0.900 | 1.000 | 0.100 | -0.100 | 0.036336 | no |
| hf_distilgpt2 | pp_singular_attractor_beside | 1.000 | 0.900 | -0.100 | 0.100 | -0.043104 | yes |
| hf_distilgpt2 | pp_singular_attractor_near | 0.900 | 1.000 | 0.100 | -0.100 | 0.185626 | no |
| hf_distilgpt2 | pp_singular_attractor_with | 0.900 | 0.800 | -0.100 | 0.100 | 0.019414 | yes |
| hf_distilgpt2 | simple_plural | 1.000 | 1.000 | 0.000 | 0.000 | 0.052598 | no |
| hf_distilgpt2 | simple_singular | 0.725 | 0.775 | 0.050 | -0.050 | 0.081096 | no |
| hf_gpt2 | object_relative_clause_plural_embedded_that | 0.750 | 0.375 | -0.375 | 0.375 | -0.172424 | yes |
| hf_gpt2 | object_relative_clause_plural_embedded_who | 0.625 | 0.500 | -0.125 | 0.125 | -0.127123 | yes |
| hf_gpt2 | object_relative_clause_that | 0.812 | 0.750 | -0.062 | 0.062 | 0.008618 | yes |
| hf_gpt2 | object_relative_clause_who | 0.750 | 0.688 | -0.062 | 0.062 | 0.007113 | yes |
| hf_gpt2 | plural_object_relative_clause_that | 0.938 | 0.938 | 0.000 | 0.000 | 0.053532 | no |
| hf_gpt2 | pp_plural_attractor_behind | 0.500 | 0.800 | 0.300 | -0.300 | 0.124774 | no |
| hf_gpt2 | pp_plural_attractor_beside | 0.700 | 0.900 | 0.200 | -0.200 | -0.011223 | no |
| hf_gpt2 | pp_plural_attractor_near | 0.600 | 0.500 | -0.100 | 0.100 | 0.091457 | yes |
| hf_gpt2 | pp_plural_attractor_with | 0.800 | 0.400 | -0.400 | 0.400 | -0.343063 | yes |
| hf_gpt2 | pp_singular_attractor_behind | 1.000 | 1.000 | 0.000 | 0.000 | 0.006473 | no |
| hf_gpt2 | pp_singular_attractor_beside | 1.000 | 1.000 | 0.000 | 0.000 | -0.146526 | no |
| hf_gpt2 | pp_singular_attractor_near | 1.000 | 1.000 | 0.000 | 0.000 | 0.052962 | no |
| hf_gpt2 | pp_singular_attractor_with | 0.800 | 0.800 | 0.000 | 0.000 | 0.070244 | no |
| hf_gpt2 | simple_plural | 1.000 | 1.000 | 0.000 | 0.000 | 0.046316 | no |
| hf_gpt2 | simple_singular | 0.925 | 0.850 | -0.075 | 0.075 | 0.019471 | yes |
| hf_tiny_gpt2 | object_relative_clause_plural_embedded_that | 0.625 | 0.000 | -0.625 | 0.625 | -0.004529 | yes |
| hf_tiny_gpt2 | object_relative_clause_plural_embedded_who | 0.625 | 0.000 | -0.625 | 0.625 | -0.004237 | yes |
| hf_tiny_gpt2 | object_relative_clause_that | 0.375 | 0.125 | -0.250 | 0.250 | -0.001548 | yes |
| hf_tiny_gpt2 | object_relative_clause_who | 0.375 | 0.125 | -0.250 | 0.250 | -0.000467 | yes |
| hf_tiny_gpt2 | plural_object_relative_clause_that | 0.250 | 0.000 | -0.250 | 0.250 | -0.000873 | yes |
| hf_tiny_gpt2 | pp_plural_attractor_behind | 0.200 | 0.000 | -0.200 | 0.200 | -0.003222 | yes |
| hf_tiny_gpt2 | pp_plural_attractor_beside | 0.600 | 0.000 | -0.600 | 0.600 | -0.003894 | yes |
| hf_tiny_gpt2 | pp_plural_attractor_near | 0.700 | 0.000 | -0.700 | 0.700 | -0.006494 | yes |
| hf_tiny_gpt2 | pp_plural_attractor_with | 0.500 | 0.000 | -0.500 | 0.500 | -0.003418 | yes |
| hf_tiny_gpt2 | pp_singular_attractor_behind | 0.300 | 0.000 | -0.300 | 0.300 | -0.002443 | yes |
| hf_tiny_gpt2 | pp_singular_attractor_beside | 0.400 | 0.000 | -0.400 | 0.400 | -0.004877 | yes |
| hf_tiny_gpt2 | pp_singular_attractor_near | 0.400 | 0.000 | -0.400 | 0.400 | -0.001672 | yes |
| hf_tiny_gpt2 | pp_singular_attractor_with | 0.400 | 0.000 | -0.400 | 0.400 | -0.004305 | yes |
| hf_tiny_gpt2 | simple_plural | 0.425 | 0.000 | -0.425 | 0.425 | -0.005598 | yes |
| hf_tiny_gpt2 | simple_singular | 0.625 | 0.000 | -0.625 | 0.625 | -0.008926 | yes |
