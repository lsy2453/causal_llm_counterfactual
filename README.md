# LLM-Guided Contrastive Causal Bayesian Optimization

This repository contains a first-version implementation of an LLM-guided
contrastive decision layer for Causal Bayesian Optimization (CBO).

The implemented method is the conservative version:

```text
Base CBO selects the intervention set.
The contrastive layer reranks candidates only within that selected intervention set.
```

This avoids directly comparing CFScore values across intervention sets with
different dimensions, domains, and posterior geometries.

## Method Scope

The current implementation computes posterior interventional contrasts:

```text
Delta(a, b) = g(a) - g(b), where g(x) = E[Y | do(X=x)].
```

It is not the strict individualized SCM counterfactual version. Strict
abduction-action-prediction support is discussed in the accompanying research
notes.

## Main Components

- `llm_contrastive_cbo/`: method package
- `llm_contrastive_cbo/benchmarks/`: ToyGraph, CompleteGraph, and PSAHealth benchmarks
- `llm_contrastive_cbo/experiments/`: benchmark runner and metrics
- `examples/run_benchmark_suite.py`: command-line experiment entrypoint
- `tests/`: smoke tests
- `README_first_version_method.md`: detailed method and experiment notes

## Install

Use your existing Python environment with `torch` and `botorch` if available.
If BoTorch is unavailable, the runner falls back to a small NumPy RBF GP.

```bash
pip install -r requirements.txt
```

## API Keys

Do not commit API keys. Set them in your shell:

```powershell
$env:KIMI_API_KEY="..."
$env:DASHSCOPE_API_KEY="..."
```

## Run Experiments

Standard ToyGraph CBO baseline:

```powershell
python -W ignore examples/run_benchmark_suite.py --benchmark toy --method standard --num_trials 5 --verbose
```

ToyGraph contrastive ablation with heuristic selector:

```powershell
python -W ignore examples/run_benchmark_suite.py --benchmark toy --method contrastive --selector heuristic --num_trials 5 --verbose
```

PSAHealth with Kimi:

```powershell
python -W ignore examples/run_benchmark_suite.py --benchmark psa --method contrastive --selector kimi --num_trials 5 --llm_log_dir runs/llm_logs/kimi --verbose
```

PSAHealth with DashScope/Qwen:

```powershell
python -W ignore examples/run_benchmark_suite.py --benchmark psa --method contrastive --selector dashscope --num_trials 5 --llm_log_dir runs/llm_logs/dashscope --verbose
```

## Tests

```powershell
python -m compileall llm_contrastive_cbo examples tests
python -c "from tests.test_first_version_layer import test_near_tie_pool_delta, test_decision_layer_returns_near_tie_candidate; test_near_tie_pool_delta(); test_decision_layer_returns_near_tie_candidate(); print('decision layer tests passed')"
python -c "from tests.test_benchmarks import test_benchmark_do_means_are_finite, test_first_version_runner_smoke; test_benchmark_do_means_are_finite(); test_first_version_runner_smoke(); print('benchmark tests passed')"
```

## Notes

Runtime logs are written under `runs/` and are intentionally ignored by git.
They may contain model prompts and responses for experiment auditing.
