"""Optuna-backed persistent sampling with a lightweight internal fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .search_space import SearchSpace
from .types import ObjectiveSpec

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:  # pragma: no cover - optional fallback exercised only without Optuna
    optuna = None
    TrialState = None


@dataclass
class AskedCandidate:
    config: dict[str, Any]
    trial_number: int
    handle: Any = None


class StudyCoordinator:
    def __init__(
        self,
        *,
        database: Path,
        study_name: str,
        sampler_name: str,
        objectives: list[ObjectiveSpec],
        search_space: SearchSpace,
        seed: int,
        explicit_configs: list[dict[str, Any]] | None = None,
        constant_liar: bool = False,
    ) -> None:
        self.search_space = search_space
        self.sampler_name = sampler_name
        self._fallback_index = 0
        self._fallback_candidates: list[dict[str, Any]] = []
        enabled = [objective for objective in objectives if objective.enabled]
        self.metric_names = [objective.name for objective in enabled]
        self.directions = [objective.direction for objective in enabled]
        if sampler_name in {"grid", "explicit"}:
            self._fallback_candidates = (
                list(explicit_configs or []) if sampler_name == "explicit" else search_space.grid()
            )
        if optuna is None:
            self.study = None
            if sampler_name in {"tpe", "random"}:
                self._fallback_candidates = search_space.sample_random(10_000, seed)
            return
        if sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(
                seed=seed,
                multivariate=False,
                constant_liar=constant_liar,
            )
        else:
            sampler = optuna.samplers.RandomSampler(seed=seed)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{database.resolve()}",
            sampler=sampler,
            directions=self.directions,
            load_if_exists=True,
        )
        if sampler_name in {"grid", "explicit"} and not self.study.trials:
            for index in range(len(self._fallback_candidates)):
                self.study.enqueue_trial(
                    {"__explicit_index__": index},
                    user_attrs={"explicit_configuration": self._fallback_candidates[index]},
                    skip_if_exists=True,
                )

    @property
    def completed_count(self) -> int:
        if self.study is None:
            return self._fallback_index
        assert TrialState is not None
        return sum(
            trial.state in {TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED}
            for trial in self.study.trials
        )

    def ask(self) -> AskedCandidate:
        if self.study is None:
            if self._fallback_index >= len(self._fallback_candidates):
                raise StopIteration
            index = self._fallback_index
            self._fallback_index += 1
            return AskedCandidate(dict(self._fallback_candidates[index]), index)
        trial = self.study.ask()
        if self.sampler_name in {"grid", "explicit"}:
            if not self._fallback_candidates:
                raise StopIteration
            index = int(
                trial.suggest_categorical(
                    "__explicit_index__", list(range(len(self._fallback_candidates)))
                )
            )
            config = dict(self._fallback_candidates[index])
        else:
            config = self.search_space.suggest(trial)
        return AskedCandidate(config, trial.number, trial)

    def tell(
        self,
        candidate: AskedCandidate,
        values: list[float] | None,
        *,
        failed: bool = False,
        user_attrs: dict[str, Any] | None = None,
    ) -> None:
        if self.study is None or candidate.handle is None:
            return
        for key, value in (user_attrs or {}).items():
            candidate.handle.set_user_attr(key, value)
        if failed:
            assert TrialState is not None
            self.study.tell(candidate.handle, state=TrialState.FAIL)
        else:
            self.study.tell(candidate.handle, values=values)
