from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from ..benchmarks import (
    BenchmarkResult,
    CausalBenchmark,
    CompleteGraphBenchmark,
    EvaluationRecord,
    InterventionSet,
    PSAHealthBenchmark,
    ToyGraphBenchmark,
)
from ..candidate_pool import build_near_tie_pool
from ..contrast_pool import ContrastPoolBuilder
from ..explanations import LLMExplanationGenerator, TemplateExplanationGenerator
from ..llm_selectors import HeuristicContrastSelector, dashscope_contrast_selector, kimi_contrast_selector
from ..posterior import posterior_interventional_contrast
from ..scoring import ContrastScoreConfig, hard_weighted_score
from ..types import ContrastResult, InterventionCandidate
from ..validators import ContrastValidator
from .metrics import best_found_curve, explanation_fidelity_proxy, mechanism_coverage
from .surrogates import BoTorchOrNumpySurrogate, expected_improvement


@dataclass(frozen=True)
class ExperimentConfig:
    benchmark: str = "toy"
    method: str = "contrastive"
    seed: int = 0
    initial_points_per_set: int = 3
    num_trials: int = 8
    candidate_grid_points: int = 11
    optimum_grid_points: int = 41
    near_tie_delta: float = 0.02
    top_k: int = 4
    lambda_disc: float = 1.0
    lambda_unc: float = 0.15
    use_hard_weighting: bool = True
    max_historical_contrasts: int = 1
    min_local_contrasts: int = 1
    selector: str = "heuristic"
    llm_log_dir: str | None = None
    kimi_model: str = "kimi-k2.6"
    dashscope_model: str = "qwen-plus"
    verbose: bool = False
    explanation_generator: str = "template"
    explanation_log_dir: str | None = None


def get_benchmark(name: str) -> CausalBenchmark:
    key = name.lower()
    if key in {"toy", "toygraph"}:
        return ToyGraphBenchmark()
    if key in {"complete", "completegraph", "synthetic", "syntheticgraph"}:
        return CompleteGraphBenchmark()
    if key in {"psa", "health", "psahealth"}:
        return PSAHealthBenchmark()
    raise ValueError(f"Unknown benchmark: {name}")


def run_first_version_benchmark(config: ExperimentConfig) -> BenchmarkResult:
    benchmark = get_benchmark(config.benchmark)
    rng = np.random.default_rng(config.seed)
    optimum_set, optimum_x, optimum_y = benchmark.global_optimum(points_per_dim=config.optimum_grid_points)
    if config.verbose:
        _print_header(config, benchmark.name, optimum_set, optimum_x, optimum_y)

    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for intervention_set in benchmark.intervention_sets():
        xs = _initial_design(intervention_set, config.initial_points_per_set, rng)
        ys = np.array([benchmark.target_mean(intervention_set, x) for x in xs], dtype=float).reshape(-1, 1)
        data[intervention_set.name] = (xs, ys)
    initial_best_y = _current_best_y(data, benchmark.task)

    records: List[EvaluationRecord] = []
    all_contrast_results = []
    explanations = []

    for step in range(config.num_trials):
        if config.verbose:
            print(f"\n========== Step {step + 1}/{config.num_trials} ==========")
        surrogates = {
            name: BoTorchOrNumpySurrogate(xs, ys)
            for name, (xs, ys) in data.items()
        }
        current_best_y = _current_best_y(data, benchmark.task)
        if config.verbose:
            print(f"[Current best observed do-outcome] {current_best_y:.6f}")
        candidates = []
        candidate_lookup = {}
        for intervention_set in benchmark.intervention_sets():
            grid = benchmark.grid(intervention_set, points_per_dim=config.candidate_grid_points)
            surrogate = surrogates[intervention_set.name]
            mean, variance = surrogate.predict(grid)
            acquisition = expected_improvement(mean, variance, current_best_y, task=benchmark.task)
            idx = int(np.argmax(acquisition))
            candidate = InterventionCandidate(
                x=grid[idx],
                acquisition_value=float(acquisition[idx]),
                exploration_set=intervention_set.variables,
                candidate_id=intervention_set.name,
                metadata={"intervention_set": intervention_set.name},
            )
            candidates.append(candidate)
            candidate_lookup[intervention_set.name] = intervention_set
            if config.verbose:
                backend = getattr(surrogate, "backend", "unknown")
                print(
                    "[Base CBO/EI candidate] "
                    f"set={intervention_set.name} vars={intervention_set.variables} "
                    f"x={_fmt_array(candidate.x)} EI={candidate.acquisition_value:.6f} "
                    f"surrogate={backend}"
                )

        if config.method == "standard":
            selected = max(candidates, key=lambda c: c.acquisition_value)
            selected_by = "standard_cbo"
            if config.verbose:
                print(
                    "[Standard CBO selected] "
                    f"set={selected.metadata['intervention_set']} x={_fmt_array(selected.x)} "
                    f"EI={selected.acquisition_value:.6f}"
                )
        elif config.method == "contrastive":
            base_selected = max(candidates, key=lambda c: c.acquisition_value)
            base_selected_set = candidate_lookup[base_selected.metadata["intervention_set"]]
            selected_set_candidates = _top_candidates_within_set(
                benchmark=benchmark,
                intervention_set=base_selected_set,
                surrogate=surrogates[base_selected_set.name],
                current_best_y=current_best_y,
                config=config,
            )
            near_tie = build_near_tie_pool(
                tuple(selected_set_candidates),
                delta=config.near_tie_delta,
                top_k=config.top_k,
                mode="max",
            )
            if config.verbose:
                print(
                    "[CBO selected intervention set] "
                    f"set={base_selected_set.name} x={_fmt_array(base_selected.x)} "
                    f"EI={base_selected.acquisition_value:.6f}"
                )
                print(
                    "[Within-set near-tie pool] "
                    f"set={base_selected_set.name} delta={config.near_tie_delta} top_k={config.top_k}"
                )
                for item in near_tie:
                    print(
                        "  "
                        f"candidate={item.candidate_id} "
                        f"x={_fmt_array(item.x)} EI={item.acquisition_value:.6f}"
                    )
            selected = near_tie[0]
            selected_score = -np.inf
            for candidate in near_tie:
                candidate_set = candidate_lookup[candidate.metadata["intervention_set"]]
                selector = _make_selector(config)
                pool_builder = ContrastPoolBuilder(candidate_set.space)
                validator = ContrastValidator(candidate_set.space)
                contrast_pool = pool_builder.build(
                    candidate,
                    historical_x=data[candidate_set.name][0],
                    historical_y=data[candidate_set.name][1].reshape(-1),
                    lower_cost_direction={name: -1.0 for name in candidate_set.variables},
                )
                valid_pool = validator.filter(candidate, contrast_pool)
                selected_contrasts = selector.select(
                    candidate,
                    valid_pool,
                    context=benchmark.explanation_context(),
                )
                selected_contrasts = _apply_selection_constraints(
                    valid_pool,
                    selected_contrasts,
                    max_selected=3,
                    max_historical=config.max_historical_contrasts,
                    min_local=config.min_local_contrasts,
                )
                if config.verbose:
                    print(
                        "\n[Contrast pool for candidate] "
                        f"set={candidate_set.name} x={_fmt_array(candidate.x)} "
                        f"valid={len(valid_pool)}/{len(contrast_pool)} selector={config.selector}"
                    )
                    for idx, contrast in enumerate(valid_pool):
                        marker = "*" if any(np.allclose(contrast.x, s.x) for s in selected_contrasts) else " "
                        print(
                            f"  {marker} id={idx} type={contrast.contrast_type} "
                            f"source={contrast.source} changed={contrast.changed_variables} "
                            f"x={_fmt_array(contrast.x)}"
                        )
                    if selected_contrasts:
                        print("[LLM/selector selected contrasts]")
                    else:
                        print("[LLM/selector selected contrasts] none")
                candidate_results = []
                for contrast in selected_contrasts:
                    mean_delta, var_delta, p_positive = posterior_interventional_contrast(
                        surrogates[candidate_set.name],
                        candidate.x,
                        contrast.x,
                    )
                    score_config = ContrastScoreConfig(
                        lambda_disc=config.lambda_disc,
                        lambda_unc=config.lambda_unc,
                        use_hard_weighting=config.use_hard_weighting,
                    )
                    disc, unc, p_better, w_hard, w_rel, score = hard_weighted_score(
                        candidate.x,
                        contrast.x,
                        contrast.contrast_type,
                        mean_delta,
                        var_delta,
                        benchmark.task,
                        score_config,
                    )
                    all_contrast_results.append(
                        ContrastResult(
                            candidate=candidate,
                            contrast=contrast,
                            mean_delta=mean_delta,
                            variance_delta=var_delta,
                            probability_positive=p_positive,
                            discriminability=disc,
                            uncertainty=unc,
                            score=score,
                            probability_candidate_better=p_better,
                            hardness_weight=w_hard,
                            semantic_relevance_weight=w_rel,
                        )
                    )
                    candidate_results.append((contrast, score))
                    if config.verbose:
                        print(
                            "  "
                            f"contrast_type={contrast.contrast_type} "
                            f"x_b={_fmt_array(contrast.x)} "
                            f"mean_delta=g(a)-g(b)={mean_delta:.6f} "
                            f"var_delta={var_delta:.6f} "
                            f"P(candidate_better)={p_better:.4f} "
                            f"disc={disc:.6f} unc={unc:.6f} "
                            f"w_hard={w_hard:.4f} w_rel={w_rel:.4f} "
                            f"CFScore={score:.6f}"
                        )
                manual_candidate_score = max((score for _, score in candidate_results), default=-np.inf)
                score_to_compare = manual_candidate_score
                if not np.isfinite(score_to_compare):
                    score_to_compare = 0.0
                if config.verbose:
                    print(
                        f"[Candidate CFScore summary] set={candidate_set.name} "
                        f"score={score_to_compare:.6f}"
                    )
                if score_to_compare > selected_score:
                    selected = candidate
                    selected_score = score_to_compare
            selected_by = "contrastive_decision_layer"
            if config.verbose:
                print(
                    "\n[Contrastive layer selected] "
                    f"set={selected.metadata['intervention_set']} "
                    f"x={_fmt_array(selected.x)} "
                    f"base_EI={selected.acquisition_value:.6f} "
                    f"selected_CFScore={selected_score:.6f}"
                )
        else:
            raise ValueError(f"Unknown method: {config.method}")

        intervention_set = candidate_lookup[selected.metadata["intervention_set"]]
        y = benchmark.target_mean(intervention_set, selected.x)
        if config.verbose:
            print(
                "[do-evaluation] "
                f"g(x)=E[target | do({intervention_set.variables}={_fmt_array(selected.x)})] "
                f"= {y:.6f}"
            )
        xs, ys = data[intervention_set.name]
        data[intervention_set.name] = (
            np.vstack([xs, selected.x.reshape(1, -1)]),
            np.vstack([ys, np.array([[y]])]),
        )
        records.append(
            EvaluationRecord(
                step=step,
                intervention_set=intervention_set.name,
                x=selected.x,
                y=float(y),
                selected_by=selected_by,
            )
        )
        if config.verbose:
            curve_so_far = best_found_curve(records, benchmark.task)
            curve_with_initial = _best_found_curve_including_initial(initial_best_y, records, benchmark.task)
            print(f"[Best-found new trials only] {curve_so_far[-1]:.6f}")
            print(f"[Best-found including initial data] {curve_with_initial[-1]:.6f}")

        explanation = _make_explanation_generator(config).generate(
            selected=selected,
            intervention_space=intervention_set.space,
            contrast_results=all_contrast_results,
            context=benchmark.explanation_context(),
            task=benchmark.task,
        )
        explanations.append(explanation)
        if config.verbose:
            print("[Faithful explanation]")
            print(f"  generator={explanation.generator} fidelity_ok={explanation.fidelity_ok}")
            print(f"  {explanation.text}")

    curve_new_trials = best_found_curve(records, benchmark.task)
    curve_including_initial = _best_found_curve_including_initial(
        initial_best_y,
        records,
        benchmark.task,
    )
    best_idx = int(np.argmin([r.y for r in records]) if benchmark.task == "min" else np.argmax([r.y for r in records]))
    best_record = records[best_idx]
    extra = {
        "decision_policy": "base_cbo_selects_intervention_set__contrastive_reranks_within_set",
        "global_optimum_y_grid": float(optimum_y),
        "global_optimum_set_grid": optimum_set,
        "global_optimum_x_grid": optimum_x.tolist(),
        "mechanism_coverage": mechanism_coverage(benchmark, records),
        "explanation_fidelity_proxy": explanation_fidelity_proxy(all_contrast_results),
        "num_contrast_results": float(len(all_contrast_results)),
        "best_found_curve_new_trials_only": list(curve_new_trials),
        "best_found_curve_including_initial_data": list(curve_including_initial),
        "initial_best_y": float(initial_best_y),
        "contrast_type_distribution": _contrast_type_distribution(all_contrast_results),
        "explanation_fidelity_rate": _explanation_fidelity_rate(explanations),
        "latest_explanation": explanations[-1].text if explanations else "",
        "latest_explanation_generator": explanations[-1].generator if explanations else "",
    }
    return BenchmarkResult(
        benchmark_name=benchmark.name,
        method_name=config.method,
        best_y=float(best_record.y),
        best_x=best_record.x,
        best_intervention_set=best_record.intervention_set,
        simple_regret=benchmark.simple_regret(float(best_record.y), float(optimum_y)),
        best_found_curve=curve_new_trials,
        records=tuple(records),
        extra_metrics=extra,
    )


def _print_header(
    config: ExperimentConfig,
    benchmark_name: str,
    optimum_set: str,
    optimum_x: np.ndarray,
    optimum_y: float,
) -> None:
    print("========== First-Version Contrastive CBO ==========")
    print(f"benchmark={benchmark_name} method={config.method} selector={config.selector}")
    print(
        f"num_trials={config.num_trials} near_tie_delta={config.near_tie_delta} "
        f"top_k={config.top_k} policy=CBO-selects-set_then-CF-reranks-within-set"
    )
    print(f"grid_optimum_set={optimum_set} grid_optimum_x={_fmt_array(optimum_x)} grid_optimum_y={optimum_y:.6f}")


def _fmt_array(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _make_selector(config: ExperimentConfig):
    fallback = HeuristicContrastSelector(max_selected=3)
    if config.selector == "heuristic":
        return fallback
    if config.selector == "kimi":
        return kimi_contrast_selector(
            model=config.kimi_model,
            max_selected=3,
            log_dir=config.llm_log_dir,
            fallback_selector=fallback,
        )
    if config.selector == "dashscope":
        return dashscope_contrast_selector(
            model=config.dashscope_model,
            max_selected=3,
            log_dir=config.llm_log_dir,
            fallback_selector=fallback,
        )
    raise ValueError(f"Unknown selector: {config.selector}")


def _make_explanation_generator(config: ExperimentConfig):
    if config.explanation_generator == "template":
        return TemplateExplanationGenerator()
    if config.explanation_generator == "kimi":
        return LLMExplanationGenerator(
            kimi_contrast_selector(
                model=config.kimi_model,
                max_selected=3,
                log_dir=None,
                fallback_selector=HeuristicContrastSelector(max_selected=3),
            ),
            log_dir=config.explanation_log_dir,
        )
    if config.explanation_generator == "dashscope":
        return LLMExplanationGenerator(
            dashscope_contrast_selector(
                model=config.dashscope_model,
                max_selected=3,
                log_dir=None,
                fallback_selector=HeuristicContrastSelector(max_selected=3),
            ),
            log_dir=config.explanation_log_dir,
        )
    raise ValueError(f"Unknown explanation_generator: {config.explanation_generator}")


def _apply_selection_constraints(
    valid_pool,
    selected_contrasts,
    max_selected: int,
    max_historical: int,
    min_local: int,
):
    """Deterministic guardrail around LLM choices."""

    selected = []
    seen = set()
    historical_count = 0
    for contrast in selected_contrasts:
        key = tuple(np.round(contrast.x, 10))
        if key in seen:
            continue
        if contrast.contrast_type == "historical_comparison" and historical_count >= max_historical:
            continue
        selected.append(contrast)
        seen.add(key)
        if contrast.contrast_type == "historical_comparison":
            historical_count += 1

    local_count = sum(_is_local_contrast(item) for item in selected)
    if local_count < min_local:
        for contrast in valid_pool:
            if not _is_local_contrast(contrast):
                continue
            key = tuple(np.round(contrast.x, 10))
            if key in seen:
                continue
            selected.insert(0, contrast)
            seen.add(key)
            local_count += 1
            if local_count >= min_local:
                break

    while len(selected) > max_selected:
        drop_idx = None
        for idx in range(len(selected) - 1, -1, -1):
            if selected[idx].contrast_type == "historical_comparison":
                drop_idx = idx
                break
        if drop_idx is None:
            drop_idx = len(selected) - 1
        selected.pop(drop_idx)

    return tuple(selected)


def _is_local_contrast(contrast) -> bool:
    return contrast.contrast_type.startswith("single_variable") or contrast.source == "local"


def _initial_design(intervention_set: InterventionSet, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return ((intervention_set.space.lower_bounds + intervention_set.space.upper_bounds) / 2.0).reshape(1, -1)
    xs = rng.uniform(
        intervention_set.space.lower_bounds,
        intervention_set.space.upper_bounds,
        size=(n, intervention_set.space.dim),
    )
    return xs


def _top_candidates_within_set(
    benchmark: CausalBenchmark,
    intervention_set: InterventionSet,
    surrogate: BoTorchOrNumpySurrogate,
    current_best_y: float,
    config: ExperimentConfig,
) -> List[InterventionCandidate]:
    """Generate candidate points inside one intervention set.

    This implements the conservative Scheme B: CBO first selects the
    intervention set, then the contrastive layer reranks candidate points only
    within that fixed set.
    """

    grid = benchmark.grid(intervention_set, points_per_dim=config.candidate_grid_points)
    mean, variance = surrogate.predict(grid)
    acquisition = expected_improvement(mean, variance, current_best_y, task=benchmark.task)
    order = np.argsort(-acquisition)
    candidates = []
    for rank, idx in enumerate(order[: max(config.top_k * 3, config.top_k, 1)]):
        idx = int(idx)
        candidates.append(
            InterventionCandidate(
                x=grid[idx],
                acquisition_value=float(acquisition[idx]),
                exploration_set=intervention_set.variables,
                candidate_id=f"{intervention_set.name}_within_{rank}",
                metadata={"intervention_set": intervention_set.name, "within_set_rank": rank},
            )
        )
    return candidates


def _current_best_y(data: Dict[str, Tuple[np.ndarray, np.ndarray]], task: str) -> float:
    ys = np.concatenate([item[1].reshape(-1) for item in data.values()])
    return float(np.min(ys) if task == "min" else np.max(ys))


def _best_found_curve_including_initial(
    initial_best_y: float,
    records: List[EvaluationRecord],
    task: str,
) -> Tuple[float, ...]:
    curve = []
    best = initial_best_y
    for record in records:
        if task == "min":
            best = min(best, record.y)
        else:
            best = max(best, record.y)
        curve.append(float(best))
    return tuple(curve)


def _contrast_type_distribution(results: List[ContrastResult]) -> Dict[str, float]:
    if not results:
        return {}
    counts = Counter(result.contrast.contrast_type for result in results)
    total = float(sum(counts.values()))
    return {key: value / total for key, value in sorted(counts.items())}


def _explanation_fidelity_rate(explanations) -> float:
    if not explanations:
        return 0.0
    return float(np.mean([item.fidelity_ok for item in explanations]))
