from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from ..types import InterventionSpace
from .base import CausalBenchmark, InterventionSet


class PSAHealthBenchmark(CausalBenchmark):
    """PSA/HEALTH-style medical SCM for semantic contrast experiments.

    This is a compact, reproducible benchmark inspired by CBO healthcare graphs.
    It is designed for first-version method validation: LLM-style contrast
    selection, dose reduction contrasts, and explanation fidelity.
    """

    name = "PSAHealth"
    task = "min"
    observational_variables = (
        "Age",
        "BMI",
        "Aspirin",
        "Statin",
        "CancerRisk",
        "Inflammation",
        "PSA",
    )

    def intervention_sets(self) -> Tuple[InterventionSet, ...]:
        aspirin = InterventionSet(
            name="do_aspirin",
            variables=("Aspirin",),
            space=InterventionSpace(("Aspirin",), np.array([0.0]), np.array([1.0])),
            mechanism_tags=("anti_inflammation_path",),
            semantic_description="Aspirin dosage intervention in [0, 1].",
        )
        statin = InterventionSet(
            name="do_statin",
            variables=("Statin",),
            space=InterventionSpace(("Statin",), np.array([0.0]), np.array([1.0])),
            mechanism_tags=("lipid_cancer_risk_path",),
            semantic_description="Statin dosage intervention in [0, 1].",
        )
        both = InterventionSet(
            name="do_aspirin_statin",
            variables=("Aspirin", "Statin"),
            space=InterventionSpace(("Aspirin", "Statin"), np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            mechanism_tags=("anti_inflammation_path", "lipid_cancer_risk_path", "combination_therapy"),
            semantic_description="Joint aspirin/statin dosage intervention.",
        )
        return (aspirin, statin, both)

    def sample_observational(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        age = rng.normal(62.0, 8.0, size=n)
        bmi = rng.normal(27.0, 4.0, size=n)
        aspirin_propensity = self._sigmoid(-1.5 + 0.015 * (age - 60.0) + 0.03 * (bmi - 27.0))
        statin_propensity = self._sigmoid(-1.0 + 0.03 * (age - 60.0) + 0.04 * (bmi - 27.0))
        aspirin = rng.binomial(1, aspirin_propensity).astype(float)
        statin = rng.binomial(1, statin_propensity).astype(float)
        cancer = 0.025 * (age - 55.0) + 0.035 * (bmi - 25.0) - 0.45 * statin + rng.normal(0.0, 0.4, n)
        inflammation = 0.04 * (bmi - 25.0) - 0.35 * aspirin + rng.normal(0.0, 0.3, n)
        psa = 3.0 + 0.55 * cancer + 0.35 * inflammation - 0.25 * statin - 0.15 * aspirin + rng.normal(0.0, 0.25, n)
        return {
            "Age": age,
            "BMI": bmi,
            "Aspirin": aspirin,
            "Statin": statin,
            "CancerRisk": cancer,
            "Inflammation": inflammation,
            "PSA": psa,
        }

    def target_mean(self, intervention_set: InterventionSet, x: Sequence[float]) -> float:
        values = dict(zip(intervention_set.variables, np.asarray(x, dtype=float).reshape(-1)))
        rng = np.random.default_rng(20260616)
        n = 7000
        age = rng.normal(62.0, 8.0, size=n)
        bmi = rng.normal(27.0, 4.0, size=n)
        aspirin_propensity = self._sigmoid(-1.5 + 0.015 * (age - 60.0) + 0.03 * (bmi - 27.0))
        statin_propensity = self._sigmoid(-1.0 + 0.03 * (age - 60.0) + 0.04 * (bmi - 27.0))
        aspirin = aspirin_propensity
        statin = statin_propensity
        if "Aspirin" in values:
            aspirin = np.full(n, values["Aspirin"])
        if "Statin" in values:
            statin = np.full(n, values["Statin"])
        cancer = 0.025 * (age - 55.0) + 0.035 * (bmi - 25.0) - 0.45 * statin
        inflammation = 0.04 * (bmi - 25.0) - 0.35 * aspirin
        # A mild interaction makes combination therapy interpretable but not
        # trivially dominant for every context.
        combination_penalty = 0.08 * aspirin * statin
        psa = 3.0 + 0.55 * cancer + 0.35 * inflammation - 0.25 * statin - 0.15 * aspirin + combination_penalty
        return float(np.mean(psa))

    def mechanism_tags_for(self, intervention_set: InterventionSet, x: Sequence[float]) -> Tuple[str, ...]:
        tags = list(intervention_set.mechanism_tags)
        values = dict(zip(intervention_set.variables, np.asarray(x, dtype=float).reshape(-1)))
        if values.get("Statin", 0.0) >= 0.8:
            tags.append("high_statin_dose")
        if values.get("Aspirin", 0.0) >= 0.5:
            tags.append("aspirin_add_on")
        if values.get("Statin", 0.0) >= 0.8 and values.get("Aspirin", 0.0) <= 0.2:
            tags.append("psa_literature_optimum_pattern")
        return tuple(tags)

    def explanation_context(self) -> Dict[str, object]:
        return {
            "benchmark": self.name,
            "task": self.task,
            "target": "PSA",
            "target_objective": "minimize PSA",
            "domain": "healthcare_semantic_contrast",
            "causal_graph": {
                "Age": ["CancerRisk", "Inflammation", "Aspirin", "Statin"],
                "BMI": ["CancerRisk", "Inflammation", "Aspirin", "Statin"],
                "Aspirin": ["Inflammation", "PSA"],
                "Statin": ["CancerRisk", "PSA"],
                "CancerRisk": ["PSA"],
                "Inflammation": ["PSA"],
                "PSA": [],
            },
            "manipulable_variables": ["Aspirin", "Statin"],
            "context_variables": ["Age", "BMI"],
            "mediators": ["CancerRisk", "Inflammation"],
            "variable_descriptions": {
                "Age": "patient age; context variable, not intervened on",
                "BMI": "body mass index; context variable, not intervened on",
                "Aspirin": "aspirin dosage in [0,1]; anti-inflammatory intervention",
                "Statin": "statin dosage in [0,1]; cancer-risk/lipid-related intervention",
                "CancerRisk": "mediator affected by Statin and patient context; contributes to PSA",
                "Inflammation": "mediator affected by Aspirin and patient context; contributes to PSA",
                "PSA": "prostate-specific antigen target; lower is better",
            },
            "intervention_sets": {
                "do_aspirin": "intervene on aspirin dose only",
                "do_statin": "intervene on statin dose only",
                "do_aspirin_statin": "joint aspirin and statin dosage intervention",
            },
            "clinically_meaningful_contrast_types": [
                "statin_dose_reduction",
                "aspirin_add_on",
                "combination_therapy",
                "historical_incumbent_comparison",
            ],
            "scientific_questions": [
                "Is high statin necessary, or is a lower statin dose sufficient?",
                "Does adding aspirin provide value beyond statin through the inflammation pathway?",
                "Is combination therapy preferable to a single-drug intervention?",
                "Can a lower-dose or lower-risk alternative preserve most predicted gain?",
                "Does the selected contrast isolate the Statin -> CancerRisk -> PSA or Aspirin -> Inflammation -> PSA pathway?",
            ],
        }

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
