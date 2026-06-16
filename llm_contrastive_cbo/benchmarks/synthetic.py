from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Sequence, Tuple

import numpy as np

from ..types import InterventionSpace
from .base import CausalBenchmark, InterventionSet


class ToyGraphBenchmark(CausalBenchmark):
    """Toy SCM from the CBO codebase: X -> Z -> Y."""

    name = "ToyGraph"
    task = "min"
    observational_variables = ("X", "Z", "Y")

    def intervention_sets(self) -> Tuple[InterventionSet, ...]:
        return (
            InterventionSet(
                name="do_X",
                variables=("X",),
                space=InterventionSpace(("X",), np.array([-5.0]), np.array([5.0])),
                mechanism_tags=("upstream_mediator_shift",),
                semantic_description="Intervene on upstream X and let mediator Z respond.",
            ),
            InterventionSet(
                name="do_Z",
                variables=("Z",),
                space=InterventionSpace(("Z",), np.array([-5.0]), np.array([20.0])),
                mechanism_tags=("direct_mediator_control",),
                semantic_description="Directly intervene on mediator Z.",
            ),
        )

    def sample_observational(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        z = np.exp(-x) + rng.normal(size=n)
        y = np.cos(z) - np.exp(-z / 20.0) + rng.normal(size=n)
        return {"X": x, "Z": z, "Y": y}

    def target_mean(self, intervention_set: InterventionSet, x: Sequence[float]) -> float:
        value = float(np.asarray(x).reshape(-1)[0])
        if intervention_set.name == "do_X":
            z_mean = np.exp(-value)
            return float(np.cos(z_mean) - np.exp(-z_mean / 20.0))
        if intervention_set.name == "do_Z":
            return float(np.cos(value) - np.exp(-value / 20.0))
        raise KeyError(intervention_set.name)

    def explanation_context(self):
        return {
            "benchmark": self.name,
            "task": self.task,
            "target": "Y",
            "target_objective": "minimize Y",
            "causal_graph": {"X": ["Z"], "Z": ["Y"], "Y": []},
            "variable_descriptions": {
                "X": "upstream manipulable variable; affects Y only through mediator Z",
                "Z": "mediator and manipulable variable; directly determines target Y",
                "Y": "optimization target; lower is better",
            },
            "intervention_sets": {
                "do_X": "intervene upstream and allow mediator Z to respond",
                "do_Z": "directly intervene on mediator Z",
            },
            "scientific_questions": [
                "Is it better to manipulate the upstream cause X or directly control mediator Z?",
                "Does a contrast isolate the mediator pathway X -> Z -> Y?",
                "Does the contrast distinguish upstream intervention from direct mediator control?",
            ],
        }


class CompleteGraphBenchmark(CausalBenchmark):
    """Synthetic CompleteGraph-style SCM used in CBO experiments.

    This benchmark mirrors the structural spirit of the original CBO synthetic
    setting: hidden common causes U1/U2, mediator C, downstream variables D/E,
    and target Y. The first-version method uses only interventional means.
    """

    name = "CompleteGraph"
    task = "min"
    observational_variables = ("U1", "U2", "F", "A", "B", "C", "D", "E", "Y")

    def intervention_sets(self) -> Tuple[InterventionSet, ...]:
        ranges = self._ranges()
        specs = [
            ("do_B", ("B",), ("mediator_C_path",)),
            ("do_D", ("D",), ("direct_D_path",)),
            ("do_E", ("E",), ("direct_E_path",)),
            ("do_BD", ("B", "D"), ("mediator_C_path", "direct_D_path")),
            ("do_BE", ("B", "E"), ("mediator_C_path", "direct_E_path")),
            ("do_DE", ("D", "E"), ("direct_D_path", "direct_E_path")),
        ]
        result = []
        for name, variables, tags in specs:
            lb = np.array([ranges[v][0] for v in variables], dtype=float)
            ub = np.array([ranges[v][1] for v in variables], dtype=float)
            result.append(
                InterventionSet(
                    name=name,
                    variables=variables,
                    space=InterventionSpace(variables, lb, ub),
                    mechanism_tags=tags,
                    semantic_description=f"Intervene on {', '.join(variables)} in the synthetic CBO graph.",
                )
            )
        return tuple(result)

    def sample_observational(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        u1 = rng.normal(size=n)
        u2 = rng.normal(size=n)
        f = rng.normal(size=n)
        a = f**2 + u1 + rng.normal(size=n)
        b = u2 + rng.normal(size=n)
        c = np.exp(-b) + rng.normal(size=n)
        d = np.exp(-c) / 10.0 + rng.normal(size=n)
        e = np.cos(a) + c / 10.0 + rng.normal(size=n)
        y = np.cos(d) - d / 5.0 + np.sin(e) - e / 4.0 + u1 + np.exp(-u2) + rng.normal(size=n)
        return {"U1": u1, "U2": u2, "F": f, "A": a, "B": b, "C": c, "D": d, "E": e, "Y": y}

    def target_mean(self, intervention_set: InterventionSet, x: Sequence[float]) -> float:
        values = dict(zip(intervention_set.variables, np.asarray(x, dtype=float).reshape(-1)))
        # Monte Carlo over non-intervened exogenous variables; deterministic seed
        # keeps the benchmark stable across methods.
        rng = np.random.default_rng(20260615)
        n = 6000
        u1 = rng.normal(size=n)
        u2 = rng.normal(size=n)
        f = rng.normal(size=n)
        a = f**2 + u1
        b = u2
        if "B" in values:
            b = np.full(n, values["B"])
        c = np.exp(-b)
        d = np.exp(-c) / 10.0
        if "D" in values:
            d = np.full(n, values["D"])
        e = np.cos(a) + c / 10.0
        if "E" in values:
            e = np.full(n, values["E"])
        y = np.cos(d) - d / 5.0 + np.sin(e) - e / 4.0 + u1 + np.exp(-u2)
        return float(np.mean(y))

    def mechanism_tags_for(self, intervention_set: InterventionSet, x: Sequence[float]) -> Tuple[str, ...]:
        tags = list(intervention_set.mechanism_tags)
        arr = np.asarray(x, dtype=float).reshape(-1)
        if "B" in intervention_set.variables and arr[intervention_set.variables.index("B")] < -2.0:
            tags.append("strong_C_mediator_shift")
        if "D" in intervention_set.variables and abs(arr[intervention_set.variables.index("D")]) < 1.0:
            tags.append("near_D_optimum")
        if "E" in intervention_set.variables and 0.0 < arr[intervention_set.variables.index("E")] < 2.0:
            tags.append("E_sine_gain_region")
        return tuple(tags)

    def _ranges(self) -> OrderedDict[str, Tuple[float, float]]:
        return OrderedDict(
            [
                ("E", (-6.0, 3.0)),
                ("B", (-5.0, 4.0)),
                ("D", (-5.0, 5.0)),
                ("F", (-4.0, 4.0)),
            ]
        )

    def explanation_context(self):
        return {
            "benchmark": self.name,
            "task": self.task,
            "target": "Y",
            "target_objective": "minimize Y",
            "causal_graph": {
                "U1": ["A", "Y"],
                "U2": ["B", "Y"],
                "F": ["A"],
                "A": ["E"],
                "B": ["C"],
                "C": ["D", "E"],
                "D": ["Y"],
                "E": ["Y"],
                "Y": [],
            },
            "hidden_or_exogenous_variables": ["U1", "U2"],
            "manipulable_variables": ["B", "D", "E"],
            "non_manipulable_variables": ["U1", "U2", "F", "A", "C", "Y"],
            "variable_descriptions": {
                "B": "manipulable upstream variable; affects mediator C and target Y through downstream paths",
                "C": "mediator generated from B; affects D and E",
                "D": "manipulable downstream variable with direct effect on Y",
                "E": "manipulable downstream variable with direct effect on Y",
                "A": "non-manipulable parent of E affected by F and U1",
                "Y": "optimization target; lower is better",
            },
            "intervention_sets": {
                "do_B": "intervene on upstream B and test the C-mediated path",
                "do_D": "directly intervene on D",
                "do_E": "directly intervene on E",
                "do_BD": "jointly intervene on upstream B and downstream D",
                "do_BE": "jointly intervene on upstream B and downstream E",
                "do_DE": "jointly intervene on downstream D and E",
            },
            "scientific_questions": [
                "Does the contrast isolate the B -> C mediator path?",
                "Does the contrast compare direct downstream intervention against upstream mediator manipulation?",
                "Does a joint intervention reveal complementarity between D and E?",
                "Is a lower-magnitude intervention sufficient relative to the candidate?",
            ],
        }
