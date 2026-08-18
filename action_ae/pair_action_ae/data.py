"""Action-only RLDS input pipeline for training the PAIR Action AE."""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets.rlds.dataset import apply_trajectory_transforms
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import (
    allocate_threads,
    get_dataset_statistics,
    normalize_action_and_proprio,
    pprint_data_mixture,
    tree_map,
)


@dataclass(frozen=True)
class ActionDataConfig:
    data_root_dir: str = "/data/kaixi/dataset/libero"
    mixture: str = "libero_4_task_suites_no_noops"
    train_split: str = "train[:95%]"
    eval_split: str = "train[95%:]"
    batch_size: int = 1024
    shuffle_buffer_size: int = 256_000
    traj_transform_threads: Optional[int] = 4
    traj_read_threads: Optional[int] = 4
    balance_weights: bool = True
    include_dataset_name: bool = True


def _restructure_trajectory(
    traj: Dict[str, Any],
    *,
    name: str,
    standardize_fn,
    image_obs_keys: Dict[str, Optional[str]],
    depth_obs_keys: Dict[str, Optional[str]],
    state_obs_keys: List[Optional[str]],
    language_key: Optional[str],
    absolute_action_mask: Optional[List[bool]],
) -> Dict[str, Any]:
    """Same standardization contract as VLA-Adapter, without requiring images."""
    if standardize_fn is not None:
        traj = standardize_fn(traj)

    required_keys = {"observation", "action"}
    if language_key is not None:
        required_keys.add(language_key)
    if not all(k in traj for k in required_keys):
        raise ValueError(f"Trajectory is missing keys: {required_keys - set(traj.keys())}")

    traj_len = tf.shape(traj["action"])[0]
    old_obs = traj["observation"]
    new_obs = {}

    for new, old in image_obs_keys.items():
        new_obs[f"image_{new}"] = tf.repeat("", traj_len) if old is None else old_obs[old]
    for new, old in depth_obs_keys.items():
        new_obs[f"depth_{new}"] = tf.repeat("", traj_len) if old is None else old_obs[old]

    if state_obs_keys:
        new_obs["proprio"] = tf.concat(
            [
                tf.zeros((traj_len, 1), dtype=tf.float32)
                if key is None
                else tf.cast(old_obs[key], tf.float32)
                for key in state_obs_keys
            ],
            axis=1,
        )

    new_obs["timestep"] = tf.range(traj_len)

    task = {}
    if language_key is not None:
        task["language_instruction"] = traj.pop(language_key)

    output = {
        "observation": new_obs,
        "task": task,
        "action": tf.cast(traj["action"], tf.float32),
        "dataset_name": tf.repeat(name, traj_len),
    }

    if absolute_action_mask is not None:
        if len(absolute_action_mask) != output["action"].shape[-1]:
            raise ValueError(
                f"Length of absolute_action_mask ({len(absolute_action_mask)}) does not match "
                f"action dimension ({output['action'].shape[-1]})."
            )
        output["absolute_action_mask"] = tf.tile(
            tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
            [traj_len, 1],
        )

    return output


def _make_dataset_from_rlds_split(
    name: str,
    data_dir: str,
    *,
    split: str,
    standardize_fn=None,
    shuffle: bool = True,
    image_obs_keys: Dict[str, Optional[str]] = {},
    depth_obs_keys: Dict[str, Optional[str]] = {},
    state_obs_keys: List[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[List[bool]] = None,
    action_normalization_mask: Optional[List[bool]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    """VLA-Adapter-style RLDS loader with an explicit TFDS split string."""
    builder = tfds.builder(name, data_dir=data_dir)
    restructure = partial(
        _restructure_trajectory,
        name=name,
        standardize_fn=standardize_fn,
        image_obs_keys=image_obs_keys,
        depth_obs_keys=depth_obs_keys,
        state_obs_keys=state_obs_keys,
        language_key=language_key,
        absolute_action_mask=absolute_action_mask,
    )

    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(
            builder,
            split="all",
            shuffle=False,
            num_parallel_reads=num_parallel_reads,
        ).traj_map(restructure, num_parallel_calls)
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn) if standardize_fn is not None else "",
            ),
            save_dir=builder.data_dir,
        )

    dataset_statistics = tree_map(np.array, dataset_statistics)
    if action_normalization_mask is not None:
        if len(action_normalization_mask) != dataset_statistics["action"]["mean"].shape[-1]:
            raise ValueError(
                f"Length of action_normalization_mask ({len(action_normalization_mask)}) does not match "
                f"action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    dataset = dl.DLataset.from_rlds(
        builder,
        split=split,
        shuffle=shuffle,
        num_parallel_reads=num_parallel_reads,
    )
    dataset = dataset.traj_map(restructure, num_parallel_calls)
    dataset = dataset.traj_map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )
    return dataset, dataset_statistics


class TorchActionIterable:
    """Thin iterable wrapper that converts batched TF numpy outputs to torch tensors."""

    def __init__(
        self,
        dataset: dl.DLataset,
        *,
        include_dataset_name: bool,
        horizon: int,
        action_dim: int,
    ) -> None:
        self.dataset = dataset
        self.include_dataset_name = include_dataset_name
        self.horizon = horizon
        self.action_dim = action_dim

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for batch in self.dataset.as_numpy_iterator():
            actions = torch.from_numpy(np.array(batch["action"], dtype=np.float32, copy=True))
            if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_dim):
                raise ValueError(f"Expected action batch [B,{self.horizon},{self.action_dim}], got {tuple(actions.shape)}")

            output: Dict[str, Any] = {"actions": actions}
            if self.include_dataset_name and "dataset_name" in batch:
                names = batch["dataset_name"]
                output["dataset_name"] = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in np.asarray(names).reshape(-1)
                ]
            yield output


def _resolve_mixture(mixture: str) -> List[Tuple[str, float]]:
    if mixture in OXE_NAMED_MIXTURES:
        return OXE_NAMED_MIXTURES[mixture]
    return [(mixture, 1.0)]


def _make_action_stream(
    *,
    data_root_dir: str,
    mixture: str,
    split: str,
    batch_size: int,
    shuffle_buffer_size: int,
    train: bool,
    balance_weights: bool,
    traj_transform_threads: Optional[int],
    traj_read_threads: Optional[int],
    include_dataset_name: bool,
    horizon: int,
    action_dim: int,
) -> Tuple[TorchActionIterable, Dict[str, Any]]:
    mixture_spec = _resolve_mixture(mixture)
    per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
        Path(data_root_dir),
        mixture_spec,
        load_camera_views=(),
        load_depth=False,
        load_proprio=True,
        load_language=False,
        action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
    )

    if not per_dataset_kwargs:
        raise ValueError(f"No datasets were materialized for mixture `{mixture}`")

    all_dataset_statistics: Dict[str, Any] = {}
    dataset_sizes = []
    for dataset_kwargs in per_dataset_kwargs:
        data_kwargs = copy.deepcopy(dataset_kwargs)
        data_kwargs.pop("dataset_frame_transform_kwargs", None)
        _, dataset_statistics = _make_dataset_from_rlds_split(
            **data_kwargs,
            split=split,
            shuffle=False,
        )
        dataset_sizes.append(dataset_statistics["num_transitions"])
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    sample_weights = np.array(weights if weights else [1.0] * len(per_dataset_kwargs), dtype=np.float64)
    if balance_weights:
        sample_weights = sample_weights * np.array(dataset_sizes, dtype=np.float64)
    sample_weights = sample_weights / np.sum(sample_weights)
    pprint_data_mixture(per_dataset_kwargs, sample_weights)

    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    datasets = []
    for dataset_kwargs, threads, reads in zip(per_dataset_kwargs, threads_per_dataset, reads_per_dataset):
        data_kwargs = copy.deepcopy(dataset_kwargs)
        data_kwargs.pop("dataset_frame_transform_kwargs", None)
        dataset, _ = _make_dataset_from_rlds_split(
            **data_kwargs,
            split=split,
            shuffle=train,
            dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
            num_parallel_calls=threads,
            num_parallel_reads=reads,
        )
        if train:
            dataset = dataset.repeat()
        dataset = apply_trajectory_transforms(
            dataset,
            train=train,
            window_size=1,
            future_action_window_size=horizon - 1,
            skip_unlabeled=False,
            goal_relabeling_strategy=None,
            num_parallel_calls=threads,
        ).flatten(num_parallel_calls=threads)
        datasets.append(dataset)

    dataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)
    if train:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.with_ram_budget(1)

    return (
        TorchActionIterable(
            dataset,
            include_dataset_name=include_dataset_name,
            horizon=horizon,
            action_dim=action_dim,
        ),
        all_dataset_statistics,
    )


def make_action_iterables(config: ActionDataConfig) -> Tuple[TorchActionIterable, TorchActionIterable, Dict[str, Any]]:
    """Build train/eval action iterables and return their dataset statistics."""
    if NUM_ACTIONS_CHUNK != 8 or ACTION_DIM != 7:
        raise ValueError(
            "Action AE v1 expects LIBERO/CALVIN-style constants "
            f"NUM_ACTIONS_CHUNK=8 and ACTION_DIM=7, got {NUM_ACTIONS_CHUNK=} {ACTION_DIM=}"
        )

    stream_kwargs = dict(
        data_root_dir=config.data_root_dir,
        mixture=config.mixture,
        batch_size=config.batch_size,
        shuffle_buffer_size=config.shuffle_buffer_size,
        balance_weights=config.balance_weights,
        traj_transform_threads=config.traj_transform_threads,
        traj_read_threads=config.traj_read_threads,
        include_dataset_name=config.include_dataset_name,
        horizon=NUM_ACTIONS_CHUNK,
        action_dim=ACTION_DIM,
    )
    train_stream, stats = _make_action_stream(split=config.train_split, train=True, **stream_kwargs)
    eval_stream, _ = _make_action_stream(split=config.eval_split, train=False, **stream_kwargs)
    return train_stream, eval_stream, stats
