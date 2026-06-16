# First-Version Method Code Framework

This directory implements the conservative first version of the proposed
method for Causal Bayesian Optimization:

**posterior interventional contrastive reranking** inside a near-tie CBO
candidate pool.

It intentionally does not implement full individualized SCM counterfactual
inference. The numerical contrast is:

```text
Delta(a, b) = g(a) - g(b), where g(x) = E[Y | do(X=x)].
```

## Main Components

- `llm_contrastive_cbo.candidate_pool`: builds the near-tie candidate pool from
  standard CBO acquisition values.
- `llm_contrastive_cbo.contrast_pool`: constructs a feasible numerical contrast
  pool before any LLM selection.
- `llm_contrastive_cbo.llm_selectors`: provides a reproducible heuristic selector
  and an interface for a real LLM selector.
- `llm_contrastive_cbo.validators`: filters invalid or meaningless contrasts.
- `llm_contrastive_cbo.posterior`: computes posterior interventional contrast
  statistics.
- `llm_contrastive_cbo.reranker`: orchestrates the decision layer.
- `llm_contrastive_cbo.adapters.botorch`: adapts BoTorch models to the posterior
  interface.

## Integration Point With Standard CBO

In the original `VirgiAgl/CausalBayesianOptimization` loop, the standard CBO
code computes an acquisition value and optimizer proposal for each exploration
set, then selects the largest acquisition value. The first-version layer should
be inserted between:

```python
y_acquisition_list[s], x_new_list[s] = find_next_y_point(...)
```

and the final intervention execution.

The conservative Scheme B used in the experiments is:

1. Base CBO/EI selects the intervention set across all exploration sets.
2. Within that fixed intervention set, build several local high-EI candidates.
3. The contrastive layer reranks only those within-set candidates.

This avoids comparing CFScore values across intervention sets with different
dimensions, domains, and posterior geometries.

The helper below handles the conversion:

```python
from llm_contrastive_cbo import candidates_from_cbo_outputs, selected_source_index

# First choose the intervention set with the base CBO acquisition.
index = int(np.argmax(y_acquisition_list))
var_to_intervene = exploration_set[index]

# Then run contrastive reranking only within this selected intervention set.
```

## BoTorch Usage

```python
from llm_contrastive_cbo.adapters import BoTorchPosteriorProvider

posterior = BoTorchPosteriorProvider(model)
```

The BoTorch model must expose `model.posterior(X)` with posterior mean and
covariance matrix. This is sufficient for computing `g(a)-g(b)` uncertainty.

## Run Minimal Example

```bash
python examples/run_first_version_synthetic.py
```

## Benchmark Experiments

The first-version benchmark suite now includes:

- `toy`: ToyGraph-style `X -> Z -> Y` synthetic SCM.
- `complete`: CompleteGraph-style synthetic CBO benchmark with hidden common
  causes, mediator `C`, downstream `D/E`, and intervention sets
  `{B}`, `{D}`, `{E}`, `{B,D}`, `{B,E}`, `{D,E}`.
- `psa`: PSA/HEALTH-style medical semantic benchmark with aspirin/statin
  dosage interventions.

Run a smoke experiment:

```bash
python examples/run_benchmark_suite.py --benchmark toy --method contrastive
python examples/run_benchmark_suite.py --benchmark complete --method contrastive
python examples/run_benchmark_suite.py --benchmark psa --method contrastive
```

Compare LLM contrast selectors:

```bash
# Kimi / Moonshot. Requires KIMI_API_KEY in your environment.
python examples/run_benchmark_suite.py --benchmark psa --method contrastive --selector kimi --llm_log_dir runs/llm_logs/kimi

# Alibaba Cloud DashScope / Qwen. Requires DASHSCOPE_API_KEY in your environment.
python examples/run_benchmark_suite.py --benchmark psa --method contrastive --selector dashscope --llm_log_dir runs/llm_logs/dashscope
```

Default models:

- Kimi: `kimi-k2.6`
- DashScope: `qwen-plus`

The code reads API keys only from environment variables and does not store them
in logs. LLM logs contain the request payload, provider response, selected
contrast ids, and possible error messages for ablation analysis.

Run the standard CBO-style baseline without contrastive reranking:

```bash
python examples/run_benchmark_suite.py --benchmark toy --method standard
```

The runner reports:

- `simple_regret`
- `best_found_curve`
- `best_intervention_set`
- `mechanism_coverage`
- `explanation_fidelity_proxy`
- `num_contrast_results`

For step-by-step debugging, add `--verbose`:

```bash
python examples/run_benchmark_suite.py --benchmark psa --method contrastive --selector kimi --num_trials 3 --llm_log_dir runs/llm_logs/kimi --verbose
```

Verbose mode prints:

- base CBO/EI candidate for each intervention set
- near-tie candidate pool
- feasible contrast pool for each near-tie candidate
- LLM-selected contrasts
- posterior interventional contrast results `g(a)-g(b)`
- discriminability, uncertainty, and CFScore
- final selected intervention and do-evaluation

The current experimental design includes the main PSAHealth fixes:

- direction-aware `P(candidate_better)` instead of raw `P(delta>0)`
- hard-contrast-weighted CFScore
- at most one historical contrast by default
- at least one local single-variable contrast by default
- `best_found_curve_new_trials_only`
- `best_found_curve_including_initial_data`
- `contrast_type_distribution`

To ablate the new scoring and constraints:

```bash
python examples/run_benchmark_suite.py --benchmark psa --method contrastive --selector kimi --disable_hard_weighting
python examples/run_benchmark_suite.py --benchmark psa --method contrastive --selector kimi --max_historical_contrasts 3 --min_local_contrasts 0
```
