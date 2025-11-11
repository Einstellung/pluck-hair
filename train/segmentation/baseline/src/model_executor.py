"""Inference utility for trained segmentation models.

This script loads the most recent Lightning checkpoint, runs the model on
user-provided images, and writes out the predicted masks (and overlays).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "inference"

os.environ.setdefault("TORCH_HOME", str(MODELS_DIR))
os.environ.setdefault("FASTAI_HOME", str(MODELS_DIR / "fastai"))
os.environ.setdefault("SMP_CACHE_DIR", str(MODELS_DIR / "smp"))
torch.hub.set_dir(str(MODELS_DIR))

@dataclass
class ModelConfig:
    architecture: str = "unet"
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = None
    in_channels: int = 3
    classes: int = 3
    activation: Optional[str] = None


@dataclass
class DataConfig:
    image_size: int = 256


@dataclass
class PathsConfig:
    output_dir: str = "outputs"
    checkpoint_dir: Optional[str] = None


def load_config(path: Path) -> Tuple[ModelConfig, DataConfig, PathsConfig]:
    with path.open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)

    model_cfg_raw = cfg.get("model", {})
    data_cfg_raw = cfg.get("data", {})
    paths_cfg_raw = cfg.get("paths", {})

    model_kwargs = {
        field.name: model_cfg_raw.get(field.name, field.default)
        for field in ModelConfig.__dataclass_fields__.values()
    }
    data_kwargs = {
        field.name: data_cfg_raw.get(field.name, field.default)
        for field in DataConfig.__dataclass_fields__.values()
    }
    paths_kwargs = {
        field.name: paths_cfg_raw.get(field.name, field.default)
        for field in PathsConfig.__dataclass_fields__.values()
    }

    model_cfg = ModelConfig(**model_kwargs)
    data_cfg = DataConfig(**data_kwargs)
    paths_cfg = PathsConfig(**paths_kwargs)

    if not paths_cfg.checkpoint_dir:
        paths_cfg.checkpoint_dir = str(MODELS_DIR / "checkpoints")

    return model_cfg, data_cfg, paths_cfg


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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(cfg: ModelConfig) -> torch.nn.Module:
    architecture_key = _normalize_architecture_name(cfg.architecture)
    factory = SMP_ARCHITECTURE_FACTORIES.get(architecture_key)
    if factory is None:
        supported = ", ".join(sorted(SMP_ARCHITECTURE_FACTORIES.keys()))
        raise ValueError(f"Unsupported architecture '{cfg.architecture}'. Available options: {supported or '[]'}.")

    model = factory(
        encoder_name=cfg.encoder_name,
        encoder_weights=None,
        in_channels=cfg.in_channels,
        classes=cfg.classes,
        activation=None,
    )
    return model


VAL_LOSS_PATTERN = re.compile(r"val_loss=(\d+\.\d+)")
EPOCH_PATTERN = re.compile(r"epoch=(\d+)")


def _score_checkpoint(path: Path) -> Tuple[int, float, float]:
    stat = path.stat()
    mtime = stat.st_mtime
    val_match = VAL_LOSS_PATTERN.search(path.name)
    if val_match:
        val_loss = float(val_match.group(1))
        epoch_match = EPOCH_PATTERN.search(path.name)
        epoch = -float(epoch_match.group(1)) if epoch_match else 0.0
        return (0, val_loss, epoch if epoch else 0.0)
    return (1, -mtime, 0.0)


def discover_checkpoints(search_dirs: Sequence[Optional[Path]]) -> List[Path]:
    checkpoints: List[Path] = []
    seen: set[str] = set()
    for directory in search_dirs:
        if directory is None:
            continue
        if not directory.exists():
            continue
        for path in directory.glob("*.ckpt"):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            checkpoints.append(path)
    return checkpoints


def select_checkpoint(search_dirs: Sequence[Optional[Path]]) -> Path:
    candidates = discover_checkpoints(search_dirs)
    if not candidates:
        raise FileNotFoundError("Could not find any checkpoints in the configured directories.")
    candidates.sort(key=_score_checkpoint)
    return candidates[0]


def load_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    prefix = "model."
    model_state = {}
    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            model_state[key[len(prefix) :]] = tensor
    model.load_state_dict(model_state, strict=False)
    return state_dict


def preprocess_image(image: Image.Image, image_size: Optional[int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    original_size = image.size  # (width, height)

    transforms: List = []
    if image_size:
        transforms.append(T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR))
    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = T.Compose(transforms)(image).unsqueeze(0)
    return tensor, original_size


def predict_mask(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    original_size: Tuple[int, int],
    num_classes: int,
    threshold: float,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(tensor)

    if num_classes == 1:
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).float()
    else:
        mask = torch.argmax(logits, dim=1, keepdim=True).float()

    resized = F.interpolate(
        mask,
        size=(original_size[1], original_size[0]),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)


DEFAULT_COLORS = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 127, 0),
    (127, 0, 255),
    (0, 127, 255),
]


def mask_to_color(mask: np.ndarray, num_classes: int) -> np.ndarray:
    palette = DEFAULT_COLORS.copy()
    if num_classes >= len(palette):
        rng = np.random.default_rng(seed=1234)
        while len(palette) <= num_classes:
            palette.append(tuple(int(x) for x in rng.integers(0, 256, size=3)))

    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx in range(1, max(num_classes, 2)):
        color[mask == cls_idx] = palette[cls_idx % len(palette)]
    return color


def overlay_mask(image: Image.Image, color_mask: np.ndarray, alpha: float) -> Image.Image:
    overlay = Image.fromarray(color_mask, mode="RGB")
    return Image.blend(image.convert("RGB"), overlay, alpha)


def collect_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    images: List[Path] = []
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF")
    for suffix in patterns:
        images.extend(path.rglob(suffix))
    if not images:
        raise ValueError(f"No images found under: {path}")
    return sorted(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the latest trained segmentation model.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the training config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path. If omitted, the best available checkpoint is used.")
    parser.add_argument("--image", type=str, required=True, help="Image file or directory of images to segment.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where predictions are written.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on. Use 'auto' for CUDA if available.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for binary masks.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Blend factor for overlay images.")
    parser.add_argument("--no-overlay", action="store_true", help="Skip generating overlay visualisations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_config_path = Path(args.config)
    if raw_config_path.is_absolute():
        config_path = raw_config_path
    elif raw_config_path.exists():
        config_path = raw_config_path.resolve()
    else:
        config_path = (BASE_DIR / raw_config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    model_cfg, data_cfg, paths_cfg = load_config(config_path)

    device = resolve_device(args.device)

    model = build_model(model_cfg)
    model.to(device)
    model.eval()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        search_dirs: List[Optional[Path]] = []
        checkpoint_dir = Path(paths_cfg.checkpoint_dir) if paths_cfg.checkpoint_dir else None
        if checkpoint_dir is not None:
            checkpoint_dir = checkpoint_dir if checkpoint_dir.is_absolute() else BASE_DIR / checkpoint_dir
            search_dirs.append(checkpoint_dir)
        search_dirs.append(MODELS_DIR / "checkpoints")
        search_dirs.append(BASE_DIR / "outputs" / "checkpoints")
        checkpoint_path = select_checkpoint(search_dirs)

    load_weights(model, checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path}")

    input_path = Path(args.image)
    images = collect_images(input_path)

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        with Image.open(img_path) as img:
            image_rgb = img.convert("RGB")

        tensor, original_size = preprocess_image(image_rgb, data_cfg.image_size)
        tensor = tensor.to(device)

        mask = predict_mask(
            model=model,
            tensor=tensor,
            original_size=original_size,
            num_classes=model_cfg.classes,
            threshold=args.threshold,
        )

        color_mask = mask_to_color(mask, model_cfg.classes)

        mask_img = Image.fromarray(mask, mode="L")
        stem = img_path.stem
        mask_output_path = output_dir / f"{stem}_mask.png"
        mask_img.save(mask_output_path)

        if not args.no_overlay:
            overlay = overlay_mask(image_rgb, color_mask, alpha=args.alpha)
            overlay_output_path = output_dir / f"{stem}_overlay.png"
            overlay.save(overlay_output_path)
        else:
            overlay_output_path = None

        unique, counts = np.unique(mask, return_counts=True)
        stats = ", ".join(f"class {int(k)}: {int(v)} px" for k, v in zip(unique, counts))
        print(f"{img_path} -> mask: {mask_output_path}", end="")
        if overlay_output_path:
            print(f", overlay: {overlay_output_path}", end="")
        print(f" | {stats}")


if __name__ == "__main__":
    main()
