"""Adapter that exposes a trained simulation policy through the shared contract."""
from __future__ import annotations
from pathlib import Path
from .decision_env import ACTION_NAMES
from .state import state_from_metrics
from ..contracts import CandidateAction, PolicyOutput
from ..policy import select_action as baseline_select

class ReinforcementLearningPolicy:
    def __init__(self, model_path: str | Path):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError("Install stable-baselines3 to use the RL policy") from exc
        path = Path(model_path)
        if not path.exists() and not path.with_suffix(".zip").exists():
            raise FileNotFoundError(f"RL model not found: {path}")
        self.model = PPO.load(str(path))
        observation_shape = getattr(
            self.model.observation_space,
            "shape",
            None,)

        if observation_shape != (10,):
            raise ValueError(
                "Incompatible PPO model observation shape "
                f"{observation_shape!r}; expected (10,). "
                "Retrain using the explicit measurement-availability masks."
            )
        self.model_path = str(path)

    def select(self, candidates: list[CandidateAction], metrics: dict) -> PolicyOutput:
        if not candidates:
            raise ValueError("policy requires at least one validated candidate")
        state = state_from_metrics(metrics)
        action_index, _ = self.model.predict(state, deterministic=True)
        requested = ACTION_NAMES[int(action_index)]
        candidate = next((c for c in candidates if c.action == requested), None)
        if candidate is None:
            fallback = baseline_select(candidates)
            return PolicyOutput(
                selected_action=fallback.selected_action,
                confidence=fallback.confidence,
                rationale=(f"RL requested {requested}, which was not among evidence-derived candidates; "
                           f"fell back to deterministic baseline. {fallback.rationale}"),
                policy_name="ppo_with_baseline_fallback",
            )
        return PolicyOutput(
            selected_action=candidate.action,
            confidence=0.55,
            rationale=(f"PPO policy selected {requested} from the evidence-derived candidate set; "
                       "confidence is a conservative fixed indicator, not a calibrated probability."),
            policy_name="ppo_simulation_policy",
        )
