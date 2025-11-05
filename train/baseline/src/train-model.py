"""Training entrypoint for configurable semantic segmentation experiments.

This script wires together data augmentation (fastai), model definition
(segmentation_models_pytorch), and the training loop (PyTorch Lightning).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TORCH_HOME", str(MODELS_DIR))
os.environ.setdefault("FASTAI_HOME", str(MODELS_DIR / "fastai"))
os.environ.setdefault("SMP_CACHE_DIR", str(MODELS_DIR / "smp"))

import torch
torch.hub.set_dir(str(MODELS_DIR))
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


@dataclass
class DataConfig:
    """
    Data configuration that gets passed to the datamodule.

    The datamodule_class specifies which datamodule to use.
    All other fields are dataset-agnostic and get passed to the datamodule's from_config() method.
    """
    datamodule_class: str  # e.g., "data_loader.voc_dataloader.VOC2012DataModule"
    root: Optional[str] = None
    dataset_name: Optional[str] = None
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    image_size: Optional[int] = None
    valid_pct: Optional[float] = None
    seed: Optional[int] = None
    pin_memory: Optional[bool] = None
    persistent_workers: Optional[bool] = None
    datamodule_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.root is not None:
            self.root = str(self.root)
        if self.dataset_name is not None:
            self.dataset_name = str(self.dataset_name)
        if self.valid_pct is not None and not 0.0 < self.valid_pct < 1.0:
            raise ValueError("data.valid_pct must be between 0 and 1.")
        self.datamodule_kwargs = dict(self.datamodule_kwargs or {})


@dataclass
class ModelConfig:
    architecture: str = "unet"
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    in_channels: int = 3
    classes: int = 3
    activation: Optional[str] = None


@dataclass
class TrainingConfig:
    max_epochs: int = 10
    learning_rate: float = 1e-3
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    precision: int | str = 32
    accelerator: str = "auto"
    devices: int | str = 1
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 50
    gradient_clip_val: Optional[float] = None
    stochastic_weight_avg: bool = False


@dataclass
class PathsConfig:
    output_dir: str = "outputs"
    checkpoint_dir: Optional[str] = None
    num_sanity_val_steps: int = 2

    def __post_init__(self) -> None:
        output_path = Path(self.output_dir)
        default_checkpoint_dir = MODELS_DIR / "checkpoints"
        checkpoint_path = Path(self.checkpoint_dir) if self.checkpoint_dir else default_checkpoint_dir
        self.output_dir = str(output_path)
        self.checkpoint_dir = str(checkpoint_path)


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    monitor: str = "val_loss"
    patience: int = 5
    mode: str = "min"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    paths: PathsConfig
    early_stopping: EarlyStoppingConfig


def _normalize_architecture_name(name: str) -> str:
    normalized = name.lower().replace("++", "plusplus").replace("+", "plus")
    for token in ("-", "_", " "):
        normalized = normalized.replace(token, "")
    return normalized


_SMP_ARCHITECTURE_FACTORIES_RAW: Dict[str, Optional[Callable[..., torch.nn.Module]]] = {
    "unet": smp.Unet,
    "unetplusplus": getattr(smp, "UnetPlusPlus", None),
    "fpn": getattr(smp, "FPN", None),
    "deeplabv3": getattr(smp, "DeepLabV3", None),
    "deeplabv3plus": getattr(smp, "DeepLabV3Plus", None),
    "linknet": getattr(smp, "Linknet", None),
    "pspnet": getattr(smp, "PSPNet", None),
    "pan": getattr(smp, "PAN", None),
    "manet": getattr(smp, "MAnet", None),
}

SMP_ARCHITECTURE_FACTORIES: Dict[str, Callable[..., torch.nn.Module]] = {
    key: factory
    for key, factory in _SMP_ARCHITECTURE_FACTORIES_RAW.items()
    if factory is not None
}


def dataclass_from_dict(cls, data: Optional[Dict[str, Any]]) -> Any:
    if data is None:
        data = {}
    field_names = {f.name for f in fields(cls)}
    init_kwargs = {name: data[name] for name in field_names if name in data}
    return cls(**init_kwargs)


def load_config(config_path: Path) -> ExperimentConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    data_cfg = dataclass_from_dict(DataConfig, raw.get("data"))
    model_cfg = dataclass_from_dict(ModelConfig, raw.get("model"))
    training_cfg = dataclass_from_dict(TrainingConfig, raw.get("training"))
    paths_cfg = dataclass_from_dict(PathsConfig, raw.get("paths"))
    early_stopping_cfg = dataclass_from_dict(EarlyStoppingConfig, raw.get("early_stopping"))

    return ExperimentConfig(
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        paths=paths_cfg,
        early_stopping=early_stopping_cfg,
    )


def build_datamodule(cfg: ExperimentConfig) -> LightningDataModule:
    """
    Instantiate the configured LightningDataModule using polymorphism.

    This function dynamically imports the datamodule class specified in the config
    and calls its from_config() factory method. The datamodule is responsible for
    extracting the necessary parameters from the config.

    Args:
        cfg: ExperimentConfig object containing data and model configuration

    Returns:
        An instance of a LightningDataModule subclass

    Raises:
        ValueError: If datamodule_class is not specified in config
        ImportError: If the specified datamodule class cannot be imported
    """
    if not cfg.data.datamodule_class:
        raise ValueError(
            "No datamodule configured. Set 'data.datamodule_class' in the config JSON. "
            "Example: 'data_loader.voc_dataloader.VOC2012DataModule'"
        )

    # Import the datamodule class
    module_path, _, class_name = cfg.data.datamodule_class.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Invalid datamodule_class '{cfg.data.datamodule_class}'. "
            "Expected format: 'module.submodule.ClassName'"
        )

    try:
        module = importlib.import_module(module_path)
        datamodule_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Failed to import '{cfg.data.datamodule_class}': {exc}"
        ) from exc

    # Use polymorphic factory method - each datamodule knows how to build itself from config
    return datamodule_cls.from_config(cfg)


class LightningSegmentationModel(LightningModule):
    """LightningModule wrapping an SMP segmentation network with configurable loss."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.cfg = config

        architecture_key = _normalize_architecture_name(self.cfg.model.architecture)
        factory = SMP_ARCHITECTURE_FACTORIES.get(architecture_key)
        if factory is None:
            supported = ", ".join(sorted(SMP_ARCHITECTURE_FACTORIES.keys()))
            raise ValueError(
                f"Unsupported architecture '{self.cfg.model.architecture}'. "
                f"Available options: {supported or '[]'}."
            )

        try:
            self.model = factory(
                encoder_name=self.cfg.model.encoder_name,
                encoder_weights=self.cfg.model.encoder_weights,
                in_channels=self.cfg.model.in_channels,
                classes=self.cfg.model.classes,
                activation=self.cfg.model.activation,
            )
        except TypeError as exc:
            raise TypeError(
                f"Failed to instantiate architecture '{self.cfg.model.architecture}' with the provided parameters."
            ) from exc

        self.learning_rate = self.cfg.training.learning_rate
        self.weight_decay = self.cfg.training.weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Any, stage: str) -> torch.Tensor:
        images, masks = batch

        if isinstance(images, torch.Tensor):
            images = images.as_subclass(torch.Tensor)
        else:
            images = torch.as_tensor(images, device=self.device)

        if isinstance(masks, torch.Tensor):
            masks = masks.as_subclass(torch.Tensor)
        else:
            masks = torch.as_tensor(masks, device=self.device)

        logits = self(images)

        if self.cfg.model.classes == 1:
            masks = masks.float().unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, masks)
            preds = (torch.sigmoid(logits) > 0.5).long()
            targets = masks.long()
        else:
            masks = masks.long().squeeze(1) if masks.ndim == 4 else masks.long()
            loss = F.cross_entropy(logits, masks)
            preds = torch.argmax(logits, dim=1)
            targets = masks

        iou = self._mean_iou(preds, targets)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mIoU", iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer_name = self.cfg.training.optimizer.lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer '{self.cfg.training.optimizer}'.")

        return optimizer

    def _mean_iou(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = self.cfg.model.classes if self.cfg.model.classes > 1 else 2
        eps = 1e-6
        preds = preds.detach().view(preds.size(0), -1)
        targets = targets.detach().view(targets.size(0), -1)

        ious = []
        for cls_idx in range(num_classes):
            pred_mask = preds == cls_idx
            target_mask = targets == cls_idx
            intersection = (pred_mask & target_mask).float().sum(dim=1)
            union = pred_mask.float().sum(dim=1) + target_mask.float().sum(dim=1) - intersection
            iou = (intersection + eps) / (union + eps)
            valid = union > 0
            if valid.any():
                ious.append(iou[valid].mean())
        if not ious:
            return torch.tensor(0.0, device=preds.device)
        return torch.stack(ious).mean()


def build_trainer(cfg: ExperimentConfig) -> Trainer:
    output_dir = Path(cfg.paths.output_dir)
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="segmentation-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if cfg.training.stochastic_weight_avg:
        callbacks.append(
            StochasticWeightAveraging(swa_lrs=cfg.training.learning_rate)
        )

    if cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.early_stopping.monitor,
                patience=cfg.early_stopping.patience,
                mode=cfg.early_stopping.mode,
            )
        )

    logger = CSVLogger(save_dir=output_dir, name="lightning-logs")

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=cfg.paths.num_sanity_val_steps,
    )
    return trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed to override config data.seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    seed = args.seed if args.seed is not None else cfg.data.seed
    seed_everything(seed)

    data_module = build_datamodule(cfg)
    model = LightningSegmentationModel(cfg)
    trainer = build_trainer(cfg)

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
