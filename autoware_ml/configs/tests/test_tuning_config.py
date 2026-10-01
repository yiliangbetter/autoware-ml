"""Composition tests for the LitePT tuning config."""

from __future__ import annotations

import hydra
from omegaconf import OmegaConf

from autoware_ml.tuning.search_space import LitePTSearchSpace
from autoware_ml.tuning.study import load_constraints, load_objectives
from autoware_ml.utils.runtime import compose_config

TUNE_CONFIG = "segmentation3d/litept/tune_voxel012_122m_t4dataset_j6gen2"


def test_tuning_config_fixes_the_budget_and_disables_early_stopping() -> None:
    cfg = compose_config(TUNE_CONFIG)
    assert cfg.trainer.max_epochs == 12
    assert cfg.callbacks.early_stopping is None
    assert cfg.callbacks.model_checkpoint.filename == "best"
    assert cfg.logger is not None
    # The debris-recall objective is a per-class validation metric, which the
    # dataset config only breaks out at test unless widened here.
    assert list(cfg.t4dataset.segmentation3d.detail_stages) == ["val", "test"]
    iou_component = cfg.t4dataset.segmentation3d.confusion_components[0]
    assert list(iou_component.detail_stages) == ["val", "test"]
    base = compose_config("segmentation3d/litept/voxel012_122m_t4dataset_j6gen2")
    assert list(base.t4dataset.segmentation3d.detail_stages) == ["test"]


def test_tuning_block_instantiates_and_references_lie_in_the_space() -> None:
    cfg = compose_config(TUNE_CONFIG)
    space = hydra.utils.instantiate(cfg.tuning.search_space)
    assert isinstance(space, LitePTSearchSpace)
    objectives = load_objectives(OmegaConf.to_container(cfg.tuning.objectives, resolve=True))
    # Run 2 maximizes mIoU alone; debris recall and deployed latency are constraints.
    assert [objective.name for objective in objectives] == ["val/seg3d/mIoU"]
    constraints = load_constraints(OmegaConf.to_container(cfg.tuning.constraints, resolve=True))
    assert constraints[0].metric == "val/seg3d/recall_debris"
    assert constraints[1].source == "latency"
    assert constraints[1].reference == "litept_tiny"
    references = OmegaConf.to_container(cfg.tuning.reference_trials, resolve=True)
    assert set(references) == {"litept_tiny"}
    for params in references.values():
        space.validate_params(params)
    # The tiny reference reproduces the shipped tiny config it inherits from.
    tiny = space.proposal_from_params(references["litept_tiny"])
    trial_cfg = compose_config(TUNE_CONFIG, list(tiny.overrides))
    base_cfg = compose_config(TUNE_CONFIG)
    for key in ("enc_channels", "enc_num_head", "enc_depths", "enc_patch_size", "enc_rope_base"):
        assert trial_cfg.model.encoder[key] == base_cfg.model.encoder[key], key
    for key in ("dec_channels", "dec_num_head", "dec_depths", "dec_patch_size"):
        assert trial_cfg.model.seg3d_head[key] == base_cfg.model.seg3d_head[key], key
