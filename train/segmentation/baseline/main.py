import io
import os
from typing import Optional, Tuple
import argparse
import numpy as np

import requests
import torch
import torchvision.transforms as T
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.models.segmentation as tv_seg


def download_image(url: str) -> Image.Image:
    """Download an image from a URL and return a PIL Image."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def preprocess_image(img: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess image for ImageNet-pretrained encoder: to tensor and normalize.

    Returns a tensor shaped [1, 3, H, W] and the original size (W, H).
    """
    original_size = img.size  # (W, H)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(img).unsqueeze(0)
    return tensor, original_size


def postprocess_mask(mask_logits: torch.Tensor, original_size: Tuple[int, int], binary: bool = True) -> Image.Image:
    """Convert raw logits to a mask PIL image resized to original size.
    
    Args:
        mask_logits: [1, C, H, W] or [1, 1, H, W] tensor
        original_size: (W, H) tuple
        binary: If True, use sigmoid+threshold for binary; else use argmax for multi-class
    """
    with torch.no_grad():
        if binary:
            prob = torch.sigmoid(mask_logits)
            mask = (prob > 0.5).float()
            mask = torch.nn.functional.interpolate(
                mask, 
                size=(original_size[1], original_size[0]), 
                mode="bilinear",
                align_corners=False
            )
            mask = mask.squeeze(0).squeeze(0).cpu()
            mask_img = (mask * 255).byte().numpy()
        else:
            # Multi-class: use argmax
            mask = torch.argmax(mask_logits, dim=1).float()  # [1, H, W]
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1),  # [1, 1, H, W]
                size=(original_size[1], original_size[0]), 
                mode="nearest"
            )
            mask = mask.squeeze(0).squeeze(0).cpu()
            
            # Convert to uint8 and scale for visualization
            # For multi-class, we want to show different classes with different intensities
            # Scale class indices to make them visible (class 0=background=black, others get different grays)
            mask_img = ((mask / mask.max() * 255) if mask.max() > 0 else mask * 255).byte().numpy()  # Scale class indices for visibility
            
        return Image.fromarray(mask_img, mode="L")


def overlay_mask_on_image(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Overlay a colored mask onto the RGB image for visualization."""
    image = image.convert("RGBA")
    
    # Convert mask to numpy for easier manipulation
    mask_array = np.array(mask)
    
    # Create a colored overlay based on mask values
    # Use different colors for different class values
    overlay_array = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
    
    # Color mapping: background=transparent, other classes=different colors
    # Class 0 (background): transparent
    # Other classes: use red with intensity based on class value
    non_bg_mask = mask_array > 0
    if np.any(non_bg_mask):
        overlay_array[non_bg_mask] = [255, 0, 0, int(255 * alpha)]  # Red for detected objects
    
    overlay = Image.fromarray(overlay_array, "RGBA")
    blended = Image.alpha_composite(image, overlay).convert("RGB")
    return blended


def load_checkpoint_if_provided(model: torch.nn.Module, ckpt: Optional[str]) -> None:
    """Load checkpoint from a local path or HTTP(S) URL if provided."""
    if not ckpt:
        return
    print(f"Loading checkpoint from: {ckpt}")
    try:
        if ckpt.startswith("http://") or ckpt.startswith("https://"):
            resp = requests.get(ckpt, timeout=60)
            resp.raise_for_status()
            state = torch.load(io.BytesIO(resp.content), map_location="cpu")
        else:
            state = torch.load(ckpt, map_location="cpu")

        # Support common formats: full state_dict or object with 'state_dict'
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print("Checkpoint loaded.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")


def run_segmentation_demo(ckpt: Optional[str] = None, use_torchvision: bool = True):
    """Run segmentation demo.
    
    Args:
        ckpt: Optional checkpoint path/URL for SMP models
        use_torchvision: If True, use torchvision's pretrained DeepLabV3 (has real weights);
                        If False, use SMP Unet (needs checkpoint or uses random weights)
    """
    # Choose a freely accessible sample image with objects/people for better segmentation visualization
    # Using an image with people and clear objects that COCO model can detect
    img_url = "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=1024"

    print("Downloading image...")
    img = download_image(img_url)
    x, original_size = preprocess_image(img)

    device = torch.device("cpu")
    
    if use_torchvision:
        # Use torchvision's pretrained DeepLabV3 with COCO weights - this has real segmentation weights!
        print("Loading torchvision DeepLabV3 (pretrained on COCO)...")
        model = tv_seg.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
        model.eval()
        model.to(device)
        x = x.to(device)
        
        print("Running inference...")
        with torch.no_grad():
            output = model(x)
            logits = output['out']  # [1, 21, H, W] for COCO classes
        
        # COCO has 21 classes (including background)
        mask_img = postprocess_mask(logits, original_size, binary=False)
        
    else:
        # Use SMP Unet (needs checkpoint for real weights)
        print("Loading SMP Unet model...")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
        load_checkpoint_if_provided(model, ckpt)
        model.eval()
        model.to(device)
        x = x.to(device)

        print("Running inference...")
        with torch.no_grad():
            logits = model(x)  # [1, 1, H, W]
        
        mask_img = postprocess_mask(logits, original_size, binary=True)

    overlay_img = overlay_mask_on_image(img, mask_img, alpha=0.4)

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, "input.jpg")
    mask_path = os.path.join(out_dir, "mask.png")
    overlay_path = os.path.join(out_dir, "overlay.jpg")

    print(f"Saving results to: {out_dir}")
    img.save(img_path)
    mask_img.save(mask_path)
    overlay_img.save(overlay_path)
    print("Done.")
    if use_torchvision:
        print("Note: Using torchvision DeepLabV3 with COCO pretrained weights - you should see real segmentation!")
    else:
        print("Note: Using SMP Unet. For best results, provide --ckpt with pretrained weights.")


def main():
    parser = argparse.ArgumentParser(description="Segmentation demo: download image and run segmentation")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["torchvision", "smp"], 
        default="torchvision",
        help="Model to use: 'torchvision' (pretrained DeepLabV3, recommended) or 'smp' (Unet from SMP)"
    )
    parser.add_argument(
        "--ckpt", 
        type=str, 
        default=None, 
        help="Optional checkpoint path or URL for SMP model weights (only used with --mode smp)"
    )
    args = parser.parse_args()

    use_torchvision = (args.mode == "torchvision")
    run_segmentation_demo(ckpt=args.ckpt, use_torchvision=use_torchvision)


if __name__ == "__main__":
    main()
