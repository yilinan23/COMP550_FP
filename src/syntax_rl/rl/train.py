"""Tabular Q-learning for the structure-level syntax RL environment."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.models.scoring import build_scorer
from syntax_rl.rl.actions import STOP, SyntaxAction
from syntax_rl.rl.environment import SyntaxEnvironment
from syntax_rl.rl.rewards import RewardConfig
from syntax_rl.rl.state import SyntaxState
from syntax_rl.utils import configure_logging, ensure_dir, get_logger, resolve_project_path, seed_everything

QTable = dict[str, dict[str, float]]

LOGGER = get_logger(__name__)


def state_to_key(state: SyntaxState) -> str:
    """Serialize interpretable structural state into a compact Q-table key."""
    fields = [
        state.template_type,
        str(state.dependency_distance),
        str(state.attractor_count),
        str(state.clause_depth),
        str(int(state.stopped)),
        str(min(state.preposition_change_count, 1)),
        str(min(state.relative_pronoun_change_count, 1)),
    ]
    return "|".join(fields)


def q_update(
    old_value: float,
    reward: float,
    next_max: float,
    learning_rate: float,
    discount: float,
) -> float:
    """Apply one tabular Q-learning update."""
    target = reward + discount * next_max
    return old_value + learning_rate * (target - old_value)


def select_epsilon_greedy_action(
    q_table: QTable,
    state: SyntaxState,
    valid_actions: list[SyntaxAction],
    epsilon: float,
    rng: random.Random,
) -> SyntaxAction:
    """Select an action with epsilon-greedy exploration."""
    if not valid_actions:
        raise ValueError("Cannot select an action from an empty valid action list.")
    if rng.random() < epsilon:
        return rng.choice(valid_actions)
    return select_greedy_action(q_table, state, valid_actions)


def select_greedy_action(q_table: QTable, state: SyntaxState, valid_actions: list[SyntaxAction]) -> SyntaxAction:
    """Select the highest-Q valid action, with stable tie-breaking."""
    state_values = q_table.get(state_to_key(state), {})
    return max(valid_actions, key=lambda action: (state_values.get(action.name, 0.0), -valid_actions.index(action)))


def train_agent(config_path: str | Path) -> dict[str, Path]:
    """Train a tabular Q-learning agent and save artifacts."""
    config_file = resolve_project_path(config_path)
    config = _load_config(config_file)
    seed = int(config.get("experiment", {}).get("seed", 42))
    seed_everything(seed)
    rng = random.Random(seed)

    output_config = config.get("outputs", {})
    log_dir = ensure_dir(output_config.get("log_dir", "outputs/logs"))
    configure_logging(log_file=log_dir / "rl_train.log")

    scorer = _build_scorer(config.get("model", {}))
    reward_config = _reward_config(config.get("reward", {}))
    environment_config = config.get("environment", {})
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})

    episodes = int(training_config.get("episodes", 100))
    learning_rate = float(training_config.get("learning_rate", 0.2))
    discount = float(training_config.get("discount", 0.9))
    epsilon_start = float(training_config.get("epsilon_start", 0.4))
    epsilon_end = float(training_config.get("epsilon_end", 0.05))
    max_steps = int(environment_config.get("max_steps", 8))
    eval_episodes = int(evaluation_config.get("episodes", 25))

    q_table: QTable = {}
    training_log: list[dict[str, Any]] = []
    for episode in range(episodes):
        epsilon = _linear_decay(epsilon_start, epsilon_end, episode, max(episodes - 1, 1))
        env = SyntaxEnvironment(scorer=scorer, reward_config=reward_config, max_steps=max_steps, seed=seed + episode)
        episode_record = _run_training_episode(
            env=env,
            q_table=q_table,
            rng=rng,
            episode=episode,
            epsilon=epsilon,
            learning_rate=learning_rate,
            discount=discount,
            max_steps=max_steps,
        )
        training_log.append(episode_record)

    trained_eval = evaluate_policy(
        q_table=q_table,
        scorer=scorer,
        reward_config=reward_config,
        episodes=eval_episodes,
        max_steps=max_steps,
        seed=seed + 10_000,
        policy_name="trained",
    )
    random_eval = evaluate_policy(
        q_table={},
        scorer=scorer,
        reward_config=reward_config,
        episodes=eval_episodes,
        max_steps=max_steps,
        seed=seed + 20_000,
        policy_name="random",
        random_policy=True,
    )
    comparison = {
        "trained": summarize_trajectories(trained_eval["trajectories"]),
        "random": summarize_trajectories(random_eval["trajectories"]),
    }

    output_dir = ensure_dir(output_config.get("training_dir", "outputs/training"))
    experiment_name = config.get("experiment", {}).get("name", "syntax_rl_search")
    paths = _write_training_outputs(
        output_dir=output_dir,
        experiment_name=experiment_name,
        q_table=q_table,
        training_log=training_log,
        trained_eval=trained_eval,
        random_eval=random_eval,
        comparison=comparison,
        config=config,
    )
    LOGGER.info("Trained Q-learning agent for %s episodes over %s states", episodes, len(q_table))
    return paths


def compare_policies(config_path: str | Path, q_values_path: str | Path | None = None) -> dict[str, Path]:
    """Compare a saved trained policy against a random policy."""
    config_file = resolve_project_path(config_path)
    config = _load_config(config_file)
    output_config = config.get("outputs", {})
    output_dir = ensure_dir(output_config.get("training_dir", "outputs/training"))
    experiment_name = config.get("experiment", {}).get("name", "syntax_rl_search")
    q_path = resolve_project_path(q_values_path) if q_values_path else output_dir / f"{experiment_name}_q_values.json"
    with q_path.open("r", encoding="utf-8") as handle:
        q_table = json.load(handle)

    seed = int(config.get("experiment", {}).get("seed", 42))
    scorer = _build_scorer(config.get("model", {}))
    reward_config = _reward_config(config.get("reward", {}))
    max_steps = int(config.get("environment", {}).get("max_steps", 8))
    eval_episodes = int(config.get("evaluation", {}).get("episodes", 25))
    trained_eval = evaluate_policy(q_table, scorer, reward_config, eval_episodes, max_steps, seed + 30_000, "trained")
    random_eval = evaluate_policy({}, scorer, reward_config, eval_episodes, max_steps, seed + 40_000, "random", True)
    comparison = {
        "trained": summarize_trajectories(trained_eval["trajectories"]),
        "random": summarize_trajectories(random_eval["trajectories"]),
    }
    return _write_comparison_outputs(output_dir, experiment_name, trained_eval, random_eval, comparison)


def evaluate_policy(
    q_table: QTable,
    scorer,
    reward_config: RewardConfig,
    episodes: int,
    max_steps: int,
    seed: int,
    policy_name: str,
    random_policy: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Roll out a greedy learned policy or random policy."""
    rng = random.Random(seed)
    trajectories: list[dict[str, Any]] = []
    final_pairs: list[dict[str, Any]] = []
    for episode in range(episodes):
        env = SyntaxEnvironment(scorer=scorer, reward_config=reward_config, max_steps=max_steps, seed=seed + episode)
        trajectory = _rollout_policy(env, q_table, rng, episode, max_steps, policy_name, random_policy)
        trajectories.append(trajectory)
        if trajectory.get("final_pair"):
            final_pairs.append(trajectory["final_pair"])
    return {"trajectories": trajectories, "final_pairs": final_pairs}


def summarize_trajectories(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize rewards, trajectory lengths, distributions, and margins."""
    if not trajectories:
        return {}
    rewards = [float(trajectory["total_reward"]) for trajectory in trajectories]
    lengths = [len(trajectory["steps"]) for trajectory in trajectories]
    final_infos = [trajectory.get("final_info", {}) for trajectory in trajectories]
    margins = [
        float(info.get("reward_breakdown", {}).get("preference_margin", 0.0))
        for info in final_infos
    ]
    failures = [
        bool(info.get("reward_breakdown", {}).get("model_failed", False))
        for info in final_infos
    ]
    near_failures = [abs(margin) <= 0.2 for margin in margins]
    pairs = [trajectory.get("final_pair") or {} for trajectory in trajectories]
    return {
        "episodes": len(trajectories),
        "average_reward": sum(rewards) / len(rewards),
        "average_trajectory_length": sum(lengths) / len(lengths),
        "template_type_distribution": dict(Counter(str(pair.get("template_type", "missing")) for pair in pairs)),
        "subtype_distribution": dict(Counter(str(pair.get("subtype", "missing")) for pair in pairs)),
        "average_preference_margin": sum(margins) / len(margins),
        "failure_rate": sum(failures) / len(failures),
        "near_failure_rate": sum(near_failures) / len(near_failures),
    }


def main() -> None:
    """CLI for training or comparing a saved policy."""
    parser = argparse.ArgumentParser(description="Train or compare a tabular syntax-RL agent.")
    parser.add_argument("--config", default="configs/rl.yaml")
    parser.add_argument("--mode", choices=["train", "compare"], default="train")
    parser.add_argument("--q-values", help="Path to a saved q_values JSON file for compare mode.")
    args = parser.parse_args()

    if args.mode == "train":
        outputs = train_agent(args.config)
    else:
        outputs = compare_policies(args.config, args.q_values)
    for label, path in outputs.items():
        print(f"Wrote {label} to {path}")


def _run_training_episode(
    env: SyntaxEnvironment,
    q_table: QTable,
    rng: random.Random,
    episode: int,
    epsilon: float,
    learning_rate: float,
    discount: float,
    max_steps: int,
) -> dict[str, Any]:
    state = env.reset()
    total_reward = 0.0
    steps: list[dict[str, Any]] = []
    final_pair: dict[str, Any] | None = None
    final_info: dict[str, Any] = {}

    for _ in range(max(1, max_steps - 1)):
        action = select_epsilon_greedy_action(q_table, state, env.valid_actions(), epsilon, rng)
        next_state, reward, done, info = env.step(action)
        _update_q_table(q_table, state, action, reward, next_state, env.valid_actions(), learning_rate, discount, done)
        total_reward += reward
        steps.append(_step_record(action.name, next_state, reward, done, info))
        state = next_state
        if done:
            final_pair = info.get("pair") if isinstance(info.get("pair"), dict) else None
            final_info = _json_safe_info(info)
            if final_pair is not None or state.stopped:
                break

    if final_pair is None and not state.stopped:
        stop_action = _stop_action(env)
        next_state, reward, done, info = env.step(stop_action)
        _update_q_table(q_table, state, stop_action, reward, next_state, env.valid_actions(), learning_rate, discount, done)
        total_reward += reward
        steps.append(_step_record(STOP, next_state, reward, done, info))
        final_pair = info.get("pair") if isinstance(info.get("pair"), dict) else None
        final_info = _json_safe_info(info)

    if final_pair is not None:
        final_pair = dict(final_pair)
        final_pair["uid"] = f"train-{episode}-{final_pair['uid']}"

    return {
        "episode": episode,
        "epsilon": epsilon,
        "total_reward": total_reward,
        "trajectory_length": len(steps),
        "final_template_type": (final_pair or {}).get("template_type"),
        "final_subtype": (final_pair or {}).get("subtype"),
        "final_preference_margin": final_info.get("reward_breakdown", {}).get("preference_margin"),
        "model_failed": final_info.get("reward_breakdown", {}).get("model_failed"),
        "steps": steps,
    }


def _rollout_policy(
    env: SyntaxEnvironment,
    q_table: QTable,
    rng: random.Random,
    episode: int,
    max_steps: int,
    policy_name: str,
    random_policy: bool,
) -> dict[str, Any]:
    state = env.reset()
    total_reward = 0.0
    steps: list[dict[str, Any]] = []
    final_pair: dict[str, Any] | None = None
    final_info: dict[str, Any] = {}
    for _ in range(max(1, max_steps - 1)):
        valid_actions = env.valid_actions()
        action = rng.choice(valid_actions) if random_policy else select_greedy_action(q_table, state, valid_actions)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps.append(_step_record(action.name, next_state, reward, done, info))
        state = next_state
        if done:
            final_pair = info.get("pair") if isinstance(info.get("pair"), dict) else None
            final_info = _json_safe_info(info)
            if final_pair is not None or state.stopped:
                break

    if final_pair is None and not state.stopped:
        next_state, reward, done, info = env.step(_stop_action(env))
        total_reward += reward
        steps.append(_step_record(STOP, next_state, reward, done, info))
        final_pair = info.get("pair") if isinstance(info.get("pair"), dict) else None
        final_info = _json_safe_info(info)

    if final_pair is not None:
        final_pair = dict(final_pair)
        final_pair["uid"] = f"{policy_name}-{episode}-{final_pair['uid']}"

    return {
        "episode": episode,
        "policy": policy_name,
        "total_reward": total_reward,
        "steps": steps,
        "final_pair": final_pair,
        "final_info": final_info,
        "final_reward_breakdown": final_info.get("reward_breakdown", {}),
    }


def _update_q_table(
    q_table: QTable,
    state: SyntaxState,
    action: SyntaxAction,
    reward: float,
    next_state: SyntaxState,
    next_valid_actions: list[SyntaxAction],
    learning_rate: float,
    discount: float,
    done: bool,
) -> None:
    state_key = state_to_key(state)
    q_table.setdefault(state_key, {})
    old_value = q_table[state_key].get(action.name, 0.0)
    next_values = q_table.get(state_to_key(next_state), {})
    next_max = 0.0 if done else max((next_values.get(next_action.name, 0.0) for next_action in next_valid_actions), default=0.0)
    q_table[state_key][action.name] = q_update(old_value, reward, next_max, learning_rate, discount)


def _write_training_outputs(
    output_dir: Path,
    experiment_name: str,
    q_table: QTable,
    training_log: list[dict[str, Any]],
    trained_eval: dict[str, list[dict[str, Any]]],
    random_eval: dict[str, list[dict[str, Any]]],
    comparison: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "q_values": output_dir / f"{experiment_name}_q_values.json",
        "training_log": output_dir / f"{experiment_name}_training_rewards.csv",
        "training_summary": output_dir / f"{experiment_name}_training_summary.json",
        "trained_trajectories": output_dir / f"{experiment_name}_trained_trajectories.json",
        "trained_pairs": output_dir / f"{experiment_name}_trained_final_pairs.jsonl",
        "random_trajectories": output_dir / f"{experiment_name}_random_trajectories.json",
        "random_pairs": output_dir / f"{experiment_name}_random_final_pairs.jsonl",
        "comparison": output_dir / f"{experiment_name}_policy_comparison.json",
        "comparison_csv": output_dir / f"{experiment_name}_policy_comparison.csv",
    }
    _write_json(q_table, paths["q_values"])
    _write_training_log(training_log, paths["training_log"])
    _write_json({"config": config, "comparison": comparison, "states": len(q_table)}, paths["training_summary"])
    _write_json(trained_eval["trajectories"], paths["trained_trajectories"])
    _write_jsonl(trained_eval["final_pairs"], paths["trained_pairs"])
    _write_json(random_eval["trajectories"], paths["random_trajectories"])
    _write_jsonl(random_eval["final_pairs"], paths["random_pairs"])
    _write_json(comparison, paths["comparison"])
    _write_comparison_csv(comparison, paths["comparison_csv"])
    return paths


def _write_comparison_outputs(
    output_dir: Path,
    experiment_name: str,
    trained_eval: dict[str, list[dict[str, Any]]],
    random_eval: dict[str, list[dict[str, Any]]],
    comparison: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "trained_trajectories": output_dir / f"{experiment_name}_trained_trajectories.json",
        "trained_pairs": output_dir / f"{experiment_name}_trained_final_pairs.jsonl",
        "random_trajectories": output_dir / f"{experiment_name}_random_trajectories.json",
        "random_pairs": output_dir / f"{experiment_name}_random_final_pairs.jsonl",
        "comparison": output_dir / f"{experiment_name}_policy_comparison.json",
        "comparison_csv": output_dir / f"{experiment_name}_policy_comparison.csv",
    }
    _write_json(trained_eval["trajectories"], paths["trained_trajectories"])
    _write_jsonl(trained_eval["final_pairs"], paths["trained_pairs"])
    _write_json(random_eval["trajectories"], paths["random_trajectories"])
    _write_jsonl(random_eval["final_pairs"], paths["random_pairs"])
    _write_json(comparison, paths["comparison"])
    _write_comparison_csv(comparison, paths["comparison_csv"])
    return paths


def _build_scorer(model_config: dict[str, Any]):
    return build_scorer(
        provider=model_config.get("provider", "length_normalized"),
        model_name=model_config.get("name"),
        device=model_config.get("device"),
        normalize_by_token_count=model_config.get("normalize_by_token_count", True),
    )


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


def _step_record(action_name: str, state: SyntaxState, reward: float, done: bool, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": action_name,
        "state": state.as_dict(),
        "reward": reward,
        "done": done,
        "info": _json_safe_info(info),
    }


def _stop_action(env: SyntaxEnvironment) -> SyntaxAction:
    for action in env.valid_actions():
        if action.name == STOP:
            return action
    raise RuntimeError("Cannot stop from current environment state.")


def _linear_decay(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    fraction = min(max(step / total_steps, 0.0), 1.0)
    return start + fraction * (end - start)


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


def _write_training_log(training_log: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "episode",
        "epsilon",
        "total_reward",
        "trajectory_length",
        "final_template_type",
        "final_subtype",
        "final_preference_margin",
        "model_failed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in training_log:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_comparison_csv(comparison: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "policy",
        "episodes",
        "average_reward",
        "average_trajectory_length",
        "average_preference_margin",
        "failure_rate",
        "near_failure_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy, summary in comparison.items():
            row = {"policy": policy}
            row.update({field: summary.get(field) for field in fieldnames if field != "policy"})
            writer.writerow(row)


def _json_safe_info(info: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in info.items() if key != "pair" or isinstance(value, dict)}


if __name__ == "__main__":
    main()
