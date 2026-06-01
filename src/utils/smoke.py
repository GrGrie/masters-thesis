from __future__ import annotations

from typing import Any

import torch

from src.utils.config import get_output_dir, save_config_snapshot, validate_config
from src.utils.logging import get_logger, log_environment, setup_logging
from src.utils.seed import set_seed


def run_smoke_check(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config, mode="smoke")

    output_dir = get_output_dir(config) / "smoke"
    setup_logging(output_dir)
    logger = get_logger(__name__)

    seed = int(config["experiment"]["seed"])
    num_classes = int(config["model"]["num_classes"])

    set_seed(seed, deterministic=True)
    log_environment(logger)
    snapshot_path = save_config_snapshot(config, output_dir)

    batch_size = 4
    image_size = int(config.get("data", {}).get("image_size", 224))
    images = torch.randn(batch_size, 3, image_size, image_size)
    targets = torch.tensor([i % num_classes for i in range(batch_size)], dtype=torch.long)
    spurious = torch.tensor([i % 2 for i in range(batch_size)], dtype=torch.long)
    groups = targets * 2 + spurious

    logits = torch.randn(batch_size, num_classes)
    predictions = logits.argmax(dim=1)
    accuracy = predictions.eq(targets).float().mean().item()

    result = {
        "batch_shape": list(images.shape),
        "targets": targets.tolist(),
        "spurious": spurious.tolist(),
        "groups": groups.tolist(),
        "accuracy": accuracy,
        "config_snapshot": str(snapshot_path),
    }
    logger.info("smoke_check=%s", result)
    return result
