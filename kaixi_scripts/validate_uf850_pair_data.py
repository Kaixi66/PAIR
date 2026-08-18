#!/usr/bin/env python3
"""Validate UF850 RLDS semantics and emit reproducible PAIR split/contract files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


TFDS_NAME = "utokyo_xarm_pick_and_place_converted_externally_to_rlds"
TFDS_VERSION = "1.0.1"
LOGICAL_DATASET_NAME = "uf850_vr_teleop_rlds"
SUPPORTED_TASKS = {
    "press-buttons",
    "place-corn-in-bowl",
    "put-cube-in-cup",
    "stack-cups",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(SUPPORTED_TASKS), required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-checksum", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        target = root / relative.removeprefix("./")
        # CASE-Lab migration replaces the dataset card while preserving the
        # source card separately. README.md is metadata, not training payload.
        if target.name == "README.md":
            if not target.is_file():
                raise FileNotFoundError(target)
            continue
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f"Checksum mismatch: {target}")


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    manifest = read_json(root / "dataset_manifest.json")
    conversion = read_json(root / "conversion_manifest.json")
    if manifest["task"]["id"] != args.task:
        raise ValueError(f"Task mismatch: requested {args.task}, found {manifest['task']['id']}")
    if manifest["release"]["tfds_name"] != TFDS_NAME or manifest["release"]["tfds_version"] != TFDS_VERSION:
        raise ValueError("Unexpected TFDS name/version")
    if not args.skip_checksum:
        validate_checksums(root)

    builder_dir = root / TFDS_NAME / TFDS_VERSION
    builder = tfds.builder_from_directory(str(builder_dir))
    expected_episodes = int(manifest["statistics"]["episodes"])
    if builder.info.splits["train"].num_examples != expected_episodes:
        raise ValueError("TFDS episode count does not match dataset manifest")

    prompt_paths: dict[str, list[str]] = defaultdict(list)
    total_steps = 0
    gripper_values: set[float] = set()
    action_chunks = []
    state_min = np.full(6, np.inf, dtype=np.float64)
    state_max = np.full(6, -np.inf, dtype=np.float64)

    for episode in tfds.as_numpy(builder.as_dataset(split="train", shuffle_files=False)):
        source_path = episode["episode_metadata"]["file_path"].decode()
        steps = list(episode["steps"])
        if not steps:
            raise ValueError(f"Empty episode: {source_path}")
        prompts = {step["language_instruction"].decode() for step in steps}
        if len(prompts) != 1 or not next(iter(prompts)).strip():
            raise ValueError(f"Episode must contain one non-empty instruction: {source_path}")
        prompt = next(iter(prompts))
        prompt_paths[prompt].append(source_path)

        actions = np.stack([step["action"] for step in steps])
        states = np.stack([step["observation"]["end_effector_pose"] for step in steps])
        if actions.shape[1:] != (7,) or actions.dtype != np.float32 or not np.isfinite(actions).all():
            raise ValueError(f"Invalid action tensor in {source_path}: {actions.shape} {actions.dtype}")
        if states.shape[1:] != (6,) or states.dtype != np.float32 or not np.isfinite(states).all():
            raise ValueError(f"Invalid UF850 joint tensor in {source_path}: {states.shape} {states.dtype}")
        for step in steps:
            for key in ("image", "hand_image"):
                image = step["observation"][key]
                if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                    raise ValueError(f"Invalid {key} in {source_path}: {image.shape} {image.dtype}")
        episode_grippers = set(np.unique(actions[:, 6]).astype(float).tolist())
        if not episode_grippers <= {-1.0, 1.0}:
            raise ValueError(f"Invalid gripper values in {source_path}: {episode_grippers}")
        gripper_values.update(episode_grippers)
        action_chunks.append(actions)
        state_min = np.minimum(state_min, states.min(axis=0))
        state_max = np.maximum(state_max, states.max(axis=0))
        total_steps += len(steps)

    if len({path for paths in prompt_paths.values() for path in paths}) != expected_episodes:
        raise ValueError("Episode paths are not unique or complete")
    prompt_counts = Counter({prompt: len(paths) for prompt, paths in prompt_paths.items()})
    if args.task in {"put-cube-in-cup", "stack-cups"}:
        if len(prompt_counts) != 3 or set(prompt_counts.values()) != {17}:
            raise ValueError(f"Expected three balanced 17-episode instructions, got {dict(prompt_counts)}")
        validation_per_prompt = 2
    else:
        if len(prompt_counts) != 1:
            raise ValueError(f"Expected one instruction, got {dict(prompt_counts)}")
        validation_per_prompt = 5
    if total_steps != int(manifest["statistics"]["retained_steps"]):
        raise ValueError("Retained step count does not match dataset manifest")
    if gripper_values != {-1.0, 1.0}:
        raise ValueError(f"Dataset must contain both gripper states, got {gripper_values}")

    actions = np.concatenate(action_chunks, axis=0)
    recorded_stats = conversion["action_statistics"]
    for key, actual in {
        "mean": actions.mean(axis=0),
        "std": actions.std(axis=0),
        "min": actions.min(axis=0),
        "max": actions.max(axis=0),
    }.items():
        # TFDS iteration order can change float32 reduction order slightly.
        np.testing.assert_allclose(actual, np.asarray(recorded_stats[key]), rtol=1e-4, atol=2e-6)

    rng = random.Random(args.seed)
    validation_paths = []
    validation_prompt_counts = {}
    for prompt in sorted(prompt_paths):
        selected = rng.sample(sorted(prompt_paths[prompt]), validation_per_prompt)
        validation_paths.extend(selected)
        validation_prompt_counts[prompt] = len(selected)
    validation_paths = sorted(validation_paths)
    validation_set = set(validation_paths)
    all_paths = sorted(path for paths in prompt_paths.values() for path in paths)
    train_paths = [path for path in all_paths if path not in validation_set]

    split = {
        "schema_version": 1,
        "task": args.task,
        "seed": args.seed,
        "train_episode_paths": train_paths,
        "validation_episode_paths": validation_paths,
        "train_episode_count": len(train_paths),
        "validation_episode_count": len(validation_paths),
        "validation_prompt_counts": validation_prompt_counts,
    }
    contract = {
        "schema_version": 1,
        "robot": "UFACTORY UF850",
        "task": args.task,
        "logical_dataset_name": LOGICAL_DATASET_NAME,
        "tfds_name": TFDS_NAME,
        "tfds_version": TFDS_VERSION,
        "normalization": "bounds_q99 computed from train episodes only",
        "images": {
            "primary": {"source": "image", "shape": [224, 224, 3], "dtype": "uint8", "role": "external"},
            "wrist": {"source": "hand_image", "shape": [224, 224, 3], "dtype": "uint8", "role": "wrist"},
        },
        "proprio": {
            "source": "end_effector_pose",
            "standardized_name": "joint_angles",
            "shape": [6],
            "dtype": "float32",
            "semantics": [f"uf850_joint_{index}_angle_rad" for index in range(1, 7)],
            "observed_min": state_min.tolist(),
            "observed_max": state_max.tolist(),
        },
        "action": {
            "shape": [8, 7],
            "dtype": "float32",
            "semantics": ["delta_tcp_x_cm", "delta_tcp_y_cm", "delta_tcp_z_cm", "delta_roll_rad", "delta_pitch_rad", "delta_yaw_rad", "gripper_state"],
            "absolute_mask": [False, False, False, False, False, False, True],
            "normalization_mask": [True, True, True, True, True, True, False],
            "gripper": {"open": -1.0, "closed": 1.0},
            "chunk_semantics": "eight consecutive retained commands; filtered actions are not recomputed",
        },
        "language_instruction_counts": dict(sorted(prompt_counts.items())),
        "split_file": str(args.split_output.resolve()),
    }
    write_json(args.split_output, split)
    write_json(args.contract_output, contract)
    print(json.dumps({"status": "PASS", "episodes": expected_episodes, "steps": total_steps, "split": split}, indent=2))


if __name__ == "__main__":
    main()
