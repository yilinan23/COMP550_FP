"""Random-policy rollouts for debugging the syntax RL environment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.models.scoring import build_scorer
from syntax_rl.rl.actions import STOP
from syntax_rl.rl.environment import SyntaxEnvironment
from syntax_rl.rl.rewards import RewardConfig
from syntax_rl.utils import configure_logging, ensure_dir, get_logger, resolve_project_path, seed_everything

LOGGER = get_logger(__name__)


def run_random_rollouts(config_path: str | Path) -> dict[str, Path]:
    """Run random valid-action rollouts and save trajectories plus final pairs."""
    config_file = resolve_project_path(config_path)
    config = _load_config(config_file)
    seed = int(config.get("experiment", {}).get("seed", 42))
    seed_everything(seed)
    rng = random.Random(seed)

    output_config = config.get("outputs", {})
    log_dir = ensure_dir(output_config.get("log_dir", "outputs/logs"))
    configure_logging(log_file=log_dir / "rl_rollout.log")

    model_config = config.get("model", {"provider": "length_normalized"})
    scorer = build_scorer(
        provider=model_config.get("provider", "length_normalized"),
        model_name=model_config.get("name"),
        device=model_config.get("device"),
        normalize_by_token_count=model_config.get("normalize_by_token_count", True),
    )

    environment_config = config.get("environment", {})
    reward_config = _reward_config(config.get("reward", {}))
    rollout_config = config.get("rollout", {})
    episodes = int(rollout_config.get("episodes", 10))
    max_steps = int(environment_config.get("max_steps", 8))

    trajectories: list[dict[str, Any]] = []
    final_pairs: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        env = SyntaxEnvironment(
            scorer=scorer,
            reward_config=reward_config,
            max_steps=max_steps,
            seed=seed + episode_index,
        )
        trajectory = _run_one_episode(env=env, rng=rng, episode_index=episode_index, max_steps=max_steps)
        trajectories.append(trajectory)
        if trajectory.get("final_pair"):
            final_pairs.append(trajectory["final_pair"])

    output_dir = ensure_dir(output_config.get("rollout_dir", "outputs/rollouts"))
    experiment_name = config.get("experiment", {}).get("name", "syntax_rl_search")
    trajectories_path = output_dir / f"{experiment_name}_trajectories.json"
    pairs_path = output_dir / f"{experiment_name}_final_pairs.jsonl"
    _write_json(trajectories, trajectories_path)
    _write_jsonl(final_pairs, pairs_path)
    LOGGER.info("Saved %s random rollouts", len(trajectories))
    return {"trajectories": trajectories_path, "pairs": pairs_path}


def main() -> None:
    """CLI entry point for random-policy rollouts."""
    parser = argparse.ArgumentParser(description="Run random syntax-RL rollouts.")
    parser.add_argument(
        "--config",
        default="configs/rl.yaml",
        help="Path to a YAML config file, relative to the project root by default.",
    )
    args = parser.parse_args()
    outputs = run_random_rollouts(args.config)
    print(f"Wrote rollout trajectories to {outputs['trajectories']}")
    print(f"Wrote final pairs to {outputs['pairs']}")


def _run_one_episode(
    env: SyntaxEnvironment,
    rng: random.Random,
    episode_index: int,
    max_steps: int,
) -> dict[str, Any]:
    state = env.reset()
    steps: list[dict[str, Any]] = []
    done = False
    total_reward = 0.0
    final_pair: dict[str, Any] | None = None
    final_info: dict[str, Any] = {}
    final_reward_breakdown: dict[str, Any] = {}

    for _ in range(max(1, max_steps - 1)):
        valid_actions = env.valid_actions()
        action = _sample_action(valid_actions, rng)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps.append(
            {
                "action": action.name,
                "state": next_state.as_dict(),
                "reward": reward,
                "done": done,
                "info": _json_safe_info(info),
            }
        )
        state = next_state
        if done:
            final_pair = info.get("pair") if isinstance(info.get("pair"), dict) else None
            final_info = _json_safe_info(info)
            final_reward_breakdown = _reward_breakdown_from_info(final_info)
            if final_pair is not None or next_state.stopped:
                break
            done = False

    if not done and not state.stopped:
        next_state, reward, done, info = env.step(_sample_stop_action(env))
        total_reward += reward
        steps.append(
            {
                "action": STOP,
                "state": next_state.as_dict(),
                "reward": reward,
                "done": done,
                "info": _json_safe_info(info),
            }
        )
        final_pair = info.get("pair") if isinstance(info.get("pair"), dict) else None
        if final_pair is not None:
            final_pair = dict(final_pair)
            final_pair["uid"] = f"episode-{episode_index}-{final_pair['uid']}"
        final_info = _json_safe_info(info)
        final_reward_breakdown = _reward_breakdown_from_info(final_info)
    elif final_pair is not None:
        final_pair = dict(final_pair)
        final_pair["uid"] = f"episode-{episode_index}-{final_pair['uid']}"

    return {
        "episode": episode_index,
        "total_reward": total_reward,
        "steps": steps,
        "final_pair": final_pair,
        "final_info": final_info,
        "final_reward_breakdown": final_reward_breakdown,
    }


def _sample_action(valid_actions, rng: random.Random):
    non_stop = [action for action in valid_actions if action.name != STOP]
    if non_stop and rng.random() < 0.75:
        return rng.choice(non_stop)
    return rng.choice(valid_actions)


def _sample_stop_action(env: SyntaxEnvironment):
    for action in env.valid_actions():
        if action.name == STOP:
            return action
    raise RuntimeError("Cannot stop from the current state.")


def _reward_config(config: dict[str, Any]) -> RewardConfig:
    return RewardConfig(
        model_failure_weight=float(config.get("model_failure_weight", 1.0)),
        low_confidence_weight=float(config.get("low_confidence_weight", 0.6)),
        low_confidence_margin=float(config.get("low_confidence_margin", 0.2)),
        complexity_weight=float(config.get("complexity_weight", 0.02)),
        dependency_distance_weight=float(config.get("dependency_distance_weight", 0.5)),
        attractor_weight=float(config.get("attractor_weight", 1.0)),
        clause_depth_weight=float(config.get("clause_depth_weight", 1.0)),
        invalid_structure_penalty=float(config.get("invalid_structure_penalty", 1.0)),
        length_penalty=float(config.get("length_penalty", 0.002)),
        grammar_failure_penalty=float(config.get("grammar_failure_penalty", 1.0)),
        template_switch_penalty=float(config.get("template_switch_penalty", 0.15)),
        simple_stop_penalty=float(config.get("simple_stop_penalty", 0.1)),
        repeated_edit_penalty=float(config.get("repeated_edit_penalty", 0.1)),
    )


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return loaded


def _write_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in info.items() if key != "pair" or isinstance(value, dict)}


def _reward_breakdown_from_info(info: dict[str, Any]) -> dict[str, Any]:
    value = info.get("reward_breakdown")
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
