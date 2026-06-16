from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_contrastive_cbo.experiments import ExperimentConfig, run_first_version_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-version contrastive CBO benchmarks.")
    parser.add_argument("--benchmark", default="toy", choices=["toy", "complete", "psa"])
    parser.add_argument("--method", default="contrastive", choices=["standard", "contrastive"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_trials", default=6, type=int)
    parser.add_argument("--initial_points_per_set", default=3, type=int)
    parser.add_argument("--candidate_grid_points", default=9, type=int)
    parser.add_argument("--selector", default="heuristic", choices=["heuristic", "kimi", "dashscope"])
    parser.add_argument("--llm_log_dir", default=None)
    parser.add_argument("--kimi_model", default="kimi-k2.6")
    parser.add_argument("--dashscope_model", default="qwen-plus")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--explanation_generator", default="template", choices=["template", "kimi", "dashscope"])
    parser.add_argument("--explanation_log_dir", default=None)
    parser.add_argument("--disable_hard_weighting", action="store_true")
    parser.add_argument("--max_historical_contrasts", default=1, type=int)
    parser.add_argument("--min_local_contrasts", default=1, type=int)
    args = parser.parse_args()

    result = run_first_version_benchmark(
        ExperimentConfig(
            benchmark=args.benchmark,
            method=args.method,
            seed=args.seed,
            num_trials=args.num_trials,
            initial_points_per_set=args.initial_points_per_set,
            candidate_grid_points=args.candidate_grid_points,
            selector=args.selector,
            llm_log_dir=args.llm_log_dir,
            kimi_model=args.kimi_model,
            dashscope_model=args.dashscope_model,
            verbose=args.verbose,
            explanation_generator=args.explanation_generator,
            explanation_log_dir=args.explanation_log_dir,
            use_hard_weighting=not args.disable_hard_weighting,
            max_historical_contrasts=args.max_historical_contrasts,
            min_local_contrasts=args.min_local_contrasts,
        )
    )
    payload = {
        "benchmark": result.benchmark_name,
        "method": result.method_name,
        "best_y": result.best_y,
        "best_x": result.best_x.tolist(),
        "best_intervention_set": result.best_intervention_set,
        "simple_regret": result.simple_regret,
        "best_found_curve": list(result.best_found_curve),
        "extra_metrics": dict(result.extra_metrics),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
