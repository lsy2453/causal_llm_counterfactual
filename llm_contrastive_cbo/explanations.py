from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, Sequence

import json
import pathlib
import time

import numpy as np

from .llm_selectors import OpenAICompatibleContrastSelector
from .types import ContrastResult, InterventionCandidate, InterventionSpace


@dataclass(frozen=True)
class Explanation:
    """Faithful explanation generated from structured contrast evidence."""

    text: str
    evidence: Dict[str, object]
    fidelity_ok: bool
    generator: str


class ExplanationGenerator(Protocol):
    def generate(
        self,
        selected: InterventionCandidate,
        intervention_space: InterventionSpace,
        contrast_results: Sequence[ContrastResult],
        context: Mapping[str, object],
        task: str,
    ) -> Explanation:
        ...


@dataclass
class TemplateExplanationGenerator:
    """Deterministic explanation generator for reproducible experiments."""

    max_contrasts: int = 3

    def generate(
        self,
        selected: InterventionCandidate,
        intervention_space: InterventionSpace,
        contrast_results: Sequence[ContrastResult],
        context: Mapping[str, object],
        task: str,
    ) -> Explanation:
        selected_results = _results_for_selected(selected, contrast_results)
        selected_results = sorted(selected_results, key=lambda r: r.score, reverse=True)[: self.max_contrasts]
        evidence = build_explanation_evidence(selected, intervention_space, selected_results, context, task)
        if not selected_results:
            text = (
                f"The selected intervention is {evidence['candidate']['intervention']}. "
                "No valid selected contrast was available, so the explanation is limited to the base CBO decision."
            )
            return Explanation(text=text, evidence=evidence, fidelity_ok=True, generator="template")

        direction = "lower" if task == "min" else "higher"
        lines = [
            f"The selected intervention is {evidence['candidate']['intervention']}.",
            f"The target objective is to obtain a {direction} value of {evidence['target']}.",
        ]
        for item in evidence["contrasts"]:
            better = item["p_candidate_better"]
            lines.append(
                "Compared with "
                f"{item['contrast_intervention']}, the posterior interventional contrast "
                f"g(a)-g(b) has mean {item['mean_delta']:.6f} and variance {item['variance_delta']:.6f}. "
                f"For this {task} task, P(candidate better)={better:.4f}. "
                f"The contrast type is {item['contrast_type']}, with hardness weight "
                f"{item['hardness_weight']:.4f} and semantic relevance weight "
                f"{item['semantic_relevance_weight']:.4f}."
            )
        lines.append(
            "This explanation is constrained to computed posterior contrast statistics and the supplied causal graph context."
        )
        return Explanation(
            text=" ".join(lines),
            evidence=evidence,
            fidelity_ok=check_explanation_fidelity(" ".join(lines), evidence),
            generator="template",
        )


@dataclass
class LLMExplanationGenerator:
    """OpenAI-compatible LLM explanation generator with evidence constraints."""

    selector_client: OpenAICompatibleContrastSelector
    log_dir: Optional[str] = None

    def generate(
        self,
        selected: InterventionCandidate,
        intervention_space: InterventionSpace,
        contrast_results: Sequence[ContrastResult],
        context: Mapping[str, object],
        task: str,
    ) -> Explanation:
        selected_results = _results_for_selected(selected, contrast_results)
        selected_results = sorted(selected_results, key=lambda r: r.score, reverse=True)[:3]
        evidence = build_explanation_evidence(selected, intervention_space, selected_results, context, task)
        request_payload = {
            "model": self.selector_client.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate faithful scientific explanations for causal Bayesian optimization. "
                        "Use only the provided structured evidence. Do not invent numerical values, causal edges, "
                        "variables, mechanisms, or conclusions. If evidence is insufficient, say so. "
                        "Return strict JSON with fields: explanation, fidelity_notes."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "structured_evidence": evidence,
                            "required_style": (
                                "Concise academic explanation. Mention the selected intervention, the most important "
                                "contrast, mean_delta, variance_delta, P(candidate_better), and why the contrast is "
                                "mechanistically meaningful according to the provided context."
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": self.selector_client.temperature,
            "response_format": {"type": "json_object"},
            "max_tokens": 700,
        }
        api_key = self.selector_client.__dict__.get("env_api_key")
        import os

        key = os.environ.get(str(api_key))
        if not key:
            fallback = TemplateExplanationGenerator().generate(
                selected, intervention_space, contrast_results, context, task
            )
            return Explanation(
                text=fallback.text,
                evidence=evidence,
                fidelity_ok=fallback.fidelity_ok,
                generator=f"{self.selector_client.provider_name}_missing_key_template_fallback",
            )
        try:
            response = self.selector_client._post_chat_completion(key, request_payload)
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            text = str(parsed.get("explanation", "")).strip()
            if not text:
                raise ValueError("LLM explanation response did not include explanation text.")
            fidelity_ok = check_explanation_fidelity(text, evidence)
            self._write_log(request_payload, response, evidence, text, None)
            return Explanation(
                text=text,
                evidence=evidence,
                fidelity_ok=fidelity_ok,
                generator=self.selector_client.provider_name,
            )
        except Exception as exc:
            fallback = TemplateExplanationGenerator().generate(
                selected, intervention_space, contrast_results, context, task
            )
            self._write_log(request_payload, None, evidence, fallback.text, str(exc))
            return Explanation(
                text=fallback.text,
                evidence=evidence,
                fidelity_ok=fallback.fidelity_ok,
                generator=f"{self.selector_client.provider_name}_template_fallback",
            )

    def _write_log(
        self,
        request_payload: Mapping[str, object],
        response_payload: Optional[Mapping[str, object]],
        evidence: Mapping[str, object],
        explanation_text: str,
        error: Optional[str],
    ) -> None:
        if not self.log_dir:
            return
        path = pathlib.Path(self.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": self.selector_client.provider_name,
            "model": self.selector_client.model,
            "timestamp": time.time(),
            "request": request_payload,
            "response": response_payload,
            "evidence": evidence,
            "explanation": explanation_text,
            "fidelity_ok": check_explanation_fidelity(explanation_text, evidence),
            "error": error,
        }
        file_name = f"explanation_{self.selector_client.provider_name}_{int(time.time() * 1000)}.json"
        (path / file_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_explanation_evidence(
    selected: InterventionCandidate,
    intervention_space: InterventionSpace,
    selected_results: Sequence[ContrastResult],
    context: Mapping[str, object],
    task: str,
) -> Dict[str, object]:
    target = context.get("target", "target")
    return {
        "task": task,
        "target": target,
        "target_objective": context.get("target_objective", "minimize target" if task == "min" else "maximize target"),
        "causal_graph": context.get("causal_graph", {}),
        "variable_descriptions": context.get("variable_descriptions", {}),
        "scientific_questions": context.get("scientific_questions", []),
        "candidate": {
            "id": selected.candidate_id,
            "intervention": intervention_space.to_dict(selected.x),
            "acquisition_value": selected.acquisition_value,
        },
        "contrasts": [
            {
                "contrast_intervention": intervention_space.to_dict(result.contrast.x),
                "contrast_type": result.contrast.contrast_type,
                "mean_delta": result.mean_delta,
                "variance_delta": result.variance_delta,
                "p_candidate_better": result.probability_candidate_better,
                "discriminability": result.discriminability,
                "uncertainty": result.uncertainty,
                "hardness_weight": result.hardness_weight,
                "semantic_relevance_weight": result.semantic_relevance_weight,
                "cfscore": result.score,
                "rationale": result.contrast.rationale,
            }
            for result in selected_results
        ],
    }


def check_explanation_fidelity(text: str, evidence: Mapping[str, object]) -> bool:
    """Lightweight audit that the explanation uses available computed evidence."""

    if not text.strip():
        return False
    contrasts = evidence.get("contrasts", [])
    if not contrasts:
        return True
    # Require at least one computed probability or contrast statistic to be
    # reflected numerically or qualitatively in the explanation.
    for item in contrasts:
        if _contains_rounded_number(text, item.get("mean_delta")):
            return True
        if _contains_rounded_number(text, item.get("p_candidate_better")):
            return True
    return False


def _contains_rounded_number(text: str, value: object) -> bool:
    if value is None:
        return False
    try:
        val = float(value)
    except Exception:
        return False
    candidates = {f"{val:.6f}", f"{val:.4f}", f"{val:.3f}", f"{val:.2f}"}
    return any(item in text for item in candidates)


def _results_for_selected(
    selected: InterventionCandidate,
    contrast_results: Sequence[ContrastResult],
) -> Sequence[ContrastResult]:
    return [
        result
        for result in contrast_results
        if result.candidate.candidate_id == selected.candidate_id
        and np.allclose(result.candidate.x, selected.x)
    ]
