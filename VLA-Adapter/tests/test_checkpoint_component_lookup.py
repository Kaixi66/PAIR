from experiments.robot.openvla_utils import (
    checkpoint_has_component,
    find_checkpoint_file,
    load_eval_config_from_checkpoint,
)


def test_checkpoint_lookup_prefers_latest_when_multiple_exist(tmp_path):
    older = tmp_path / "pair_bridge--50000_checkpoint.pt"
    latest = tmp_path / "pair_bridge--latest_checkpoint.pt"
    older.write_bytes(b"old")
    latest.write_bytes(b"latest")

    assert checkpoint_has_component(str(tmp_path), "pair_bridge")
    assert find_checkpoint_file(str(tmp_path), "pair_bridge") == str(latest)
    assert load_eval_config_from_checkpoint(str(tmp_path))["use_pair_bridge"] is True


def test_checkpoint_lookup_prefers_highest_step_without_latest(tmp_path):
    low = tmp_path / "action_head--100_checkpoint.pt"
    high = tmp_path / "action_head--200_checkpoint.pt"
    low.write_bytes(b"low")
    high.write_bytes(b"high")

    assert find_checkpoint_file(str(tmp_path), "action_head") == str(high)
