from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import ContrastCandidate, InterventionCandidate, InterventionSpace


class LLMContrastSelector(ABC):
    """Interface for selecting contrasts from a feasible numerical pool."""

    @abstractmethod
    def select(
        self,
        candidate: InterventionCandidate,
        pool: Tuple[ContrastCandidate, ...],
        context: Mapping[str, object] | None = None,
    ) -> Tuple[ContrastCandidate, ...]:
        raise NotImplementedError


@dataclass
class HeuristicContrastSelector(LLMContrastSelector):
    """Reproducible stand-in for an LLM selector.

    Use this in the first experiments and as the non-LLM ablation. A real LLM
    implementation can keep the same interface and return a subset of ``pool``.
    """

    max_selected: int = 3
    preferred_types: Sequence[str] = (
        "dose_or_cost_reduction",
        "historical_comparison",
        "single_variable_decrease",
        "single_variable_increase",
        "uncertainty_driven_contrast",
    )

    def select(
        self,
        candidate: InterventionCandidate,
        pool: Tuple[ContrastCandidate, ...],
        context: Mapping[str, object] | None = None,
    ) -> Tuple[ContrastCandidate, ...]:
        if not pool:
            return tuple()
        type_rank: Dict[str, int] = {name: i for i, name in enumerate(self.preferred_types)}

        def key(contrast: ContrastCandidate) -> tuple[int, float]:
            rank = type_rank.get(contrast.contrast_type, len(type_rank))
            distance = float(np.linalg.norm(candidate.x - contrast.x))
            return rank, distance

        return tuple(sorted(pool, key=key)[: self.max_selected])


@dataclass
class CallbackContrastSelector(LLMContrastSelector):
    """Adapter for a concrete LLM client or prompt pipeline.

    The callback receives a serializable payload and must return selected
    contrast ids. Keeping this adapter small makes the numerical method
    independent from any particular LLM provider.
    """

    space: InterventionSpace
    callback: Callable[[Mapping[str, object]], Sequence[int]]
    max_selected: int = 3

    def select(
        self,
        candidate: InterventionCandidate,
        pool: Tuple[ContrastCandidate, ...],
        context: Mapping[str, object] | None = None,
    ) -> Tuple[ContrastCandidate, ...]:
        payload = {
            "candidate": self.space.to_dict(candidate.x),
            "candidate_id": candidate.candidate_id,
            "pool": [
                {
                    "id": i,
                    "x": self.space.to_dict(contrast.x),
                    "contrast_type": contrast.contrast_type,
                    "changed_variables": list(contrast.changed_variables),
                    "source": contrast.source,
                    "rationale": contrast.rationale,
                }
                for i, contrast in enumerate(pool)
            ],
            "context": dict(context or {}),
        }
        selected_ids = list(self.callback(payload))[: self.max_selected]
        return tuple(pool[i] for i in selected_ids if 0 <= int(i) < len(pool))


@dataclass
class OpenAICompatibleContrastSelector(LLMContrastSelector):
    """LLM selector for OpenAI-compatible chat-completions APIs.

    The selector never lets the model create new intervention values. The model
    can only return ids from the feasible numerical contrast pool.
    """

    provider_name: str
    env_api_key: str
    base_url: str
    model: str
    max_selected: int = 3
    temperature: float = 0.0
    timeout_seconds: int = 60
    log_dir: Optional[str] = None
    max_completion_tokens: int = 512
    fallback_selector: Optional[LLMContrastSelector] = None

    def select(
        self,
        candidate: InterventionCandidate,
        pool: Tuple[ContrastCandidate, ...],
        context: Mapping[str, object] | None = None,
    ) -> Tuple[ContrastCandidate, ...]:
        if not pool:
            return tuple()
        request_payload = self._request_payload(candidate, pool, context)
        api_key = os.environ.get(self.env_api_key)
        if not api_key:
            self._write_log(
                request_payload,
                None,
                [],
                error=f"missing_environment_variable:{self.env_api_key};fallback_to_heuristic",
            )
            if self.fallback_selector is not None:
                return self.fallback_selector.select(candidate, pool, context=context)
            raise RuntimeError(
                f"Missing API key environment variable {self.env_api_key}. "
                "Set it locally before using this selector."
            )

        try:
            response_payload = self._post_chat_completion(api_key, request_payload)
            content = response_payload["choices"][0]["message"]["content"]
            selected_ids = self._parse_selected_ids(content)
            self._write_log(request_payload, response_payload, selected_ids, error=None)
        except Exception as exc:
            self._write_log(request_payload, None, [], error=str(exc))
            if self.fallback_selector is not None:
                return self.fallback_selector.select(candidate, pool, context=context)
            raise

        selected = []
        seen = set()
        for item in selected_ids:
            idx = int(item)
            if 0 <= idx < len(pool) and idx not in seen:
                selected.append(pool[idx])
                seen.add(idx)
            if len(selected) >= self.max_selected:
                break
        if not selected and self.fallback_selector is not None:
            return self.fallback_selector.select(candidate, pool, context=context)
        return tuple(selected)

    def _request_payload(
        self,
        candidate: InterventionCandidate,
        pool: Tuple[ContrastCandidate, ...],
        context: Mapping[str, object] | None,
    ) -> dict:
        system_prompt = (
            "You are a scientific assistant for causal Bayesian optimization. "
            "Your only task is to select useful contrast ids from a feasible "
            "numerical contrast pool. Do not invent new interventions. Do not "
            "estimate causal effects. Prefer contrasts that isolate mechanisms, "
            "test dose/cost reduction, compare historical alternatives, or probe "
            "posterior uncertainty. Return strict JSON only."
        )
        user_payload = {
            "candidate": {
                "id": candidate.candidate_id,
                "x": candidate.x.tolist(),
                "exploration_set": list(candidate.exploration_set),
                "acquisition_value": candidate.acquisition_value,
                "metadata": candidate.metadata,
            },
            "contrast_pool": [
                {
                    "id": i,
                    "x": contrast.x.tolist(),
                    "contrast_type": contrast.contrast_type,
                    "changed_variables": list(contrast.changed_variables),
                    "source": contrast.source,
                    "rationale": contrast.rationale,
                }
                for i, contrast in enumerate(pool)
            ],
            "context": dict(context or {}),
            "required_output": {
                "selected_ids": "array of integer ids from contrast_pool",
                "rationales": "short object mapping selected id to reason",
            },
            "selection_constraints": {
                "must_select_one_local_single_variable_contrast_if_available": True,
                "must_select_at_most_one_historical_contrast": True,
                "prefer_hard_contrasts_with_similar_posterior_or_local_values": True,
                "avoid_trivial_low_performing_historical_contrasts": True,
                "do_not_select_only_historical_comparisons": True,
            },
            "max_selected": self.max_selected,
        }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_completion_tokens,
        }

    def _post_chat_completion(self, api_key: str, payload: Mapping[str, object]) -> dict:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider_name} API HTTP {exc.code}: {body}") from exc

    def _parse_selected_ids(self, content: str) -> Sequence[int]:
        parsed = json.loads(content)
        selected = parsed.get("selected_ids", [])
        if not isinstance(selected, list):
            raise ValueError("LLM response field selected_ids must be a list.")
        return [int(item) for item in selected]

    def _write_log(
        self,
        request_payload: Mapping[str, object],
        response_payload: Optional[Mapping[str, object]],
        selected_ids: Sequence[int],
        error: Optional[str],
    ) -> None:
        if not self.log_dir:
            return
        path = pathlib.Path(self.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        log_payload = {
            "provider": self.provider_name,
            "model": self.model,
            "timestamp": time.time(),
            "request": request_payload,
            "response": response_payload,
            "selected_ids": list(selected_ids),
            "error": error,
        }
        file_name = f"{self.provider_name}_{int(time.time() * 1000)}.json"
        (path / file_name).write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def kimi_contrast_selector(
    model: str = "kimi-k2.6",
    max_selected: int = 3,
    log_dir: Optional[str] = None,
    fallback_selector: Optional[LLMContrastSelector] = None,
) -> OpenAICompatibleContrastSelector:
    return OpenAICompatibleContrastSelector(
        provider_name="kimi",
        env_api_key="KIMI_API_KEY",
        base_url="https://api.moonshot.cn/v1",
        model=model,
        max_selected=max_selected,
        temperature=1.0,
        log_dir=log_dir,
        fallback_selector=fallback_selector,
    )


def dashscope_contrast_selector(
    model: str = "qwen-plus",
    max_selected: int = 3,
    log_dir: Optional[str] = None,
    fallback_selector: Optional[LLMContrastSelector] = None,
) -> OpenAICompatibleContrastSelector:
    return OpenAICompatibleContrastSelector(
        provider_name="dashscope",
        env_api_key="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model,
        max_selected=max_selected,
        log_dir=log_dir,
        fallback_selector=fallback_selector,
    )
