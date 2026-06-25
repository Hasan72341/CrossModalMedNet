import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
from src.utils.visualize import plot_translation_results
from src.utils.metrics import MetricsCalculator


def _load_state(model, ckpt):
    """Load a checkpoint, surfacing any key mismatch instead of failing silently."""
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] checkpoint key mismatch -> missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print(f"        e.g. missing: {list(missing)[:5]}")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="CrossModalMedNet Inference")
    parser.add_argument("--architecture", type=str, required=True, choices=["cyclegan", "pix2pix", "diffusion_paired", "diffusion_unpaired"])
    parser.add_argument("--region", type=str, required=True, choices=["brain", "pelvis"])
    parser.add_argument("--input", type=str, required=True, help="Path to input CT (.pt, .npy, or image)")
    parser.add_argument("--target", type=str, help="Path to target MRI (optional, for metrics)")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save result")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_model(args):
    # This logic is similar to app.py but for CLI
    # Simplified here for brevity, assuming models are already in CrossModalMedNet/models
    base_dir = Path(__file__).parent
    
    if args.architecture == "cyclegan":
        from models.cyclegan.models import build_cyclegan_2d_friendly
        model = build_cyclegan_2d_friendly(use_attention=False, use_multiscale=True)
        ckpt_path = base_dir / "checkpoints" / f"cyclegan_{args.region}" / "latest.pth"
        ckpt = torch.load(ckpt_path, map_location=args.device)
        _load_state(model, ckpt)
        model.to(args.device).eval()
        return lambda x: model.G_CT2MRI(x)

    elif args.architecture == "pix2pix":
        from models.pix2pix.models import build_pix2pix
        model = build_pix2pix(input_nc=1, output_nc=1)
        ckpt_path = base_dir / "checkpoints" / f"pix2pix_{args.region}" / "latest.pth"
        ckpt = torch.load(ckpt_path, map_location=args.device)
        _load_state(model, ckpt)
        model.to(args.device).eval()
        return lambda x: model.G_CT2MRI(x)
        
    # ... add diffusion cases similarly if needed for CLI ...
    
    raise NotImplementedError(f"Architecture {args.architecture} not yet fully supported in CLI")

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Load input
    ext = Path(args.input).suffix.lower()
    if ext == ".pt":
        img = torch.load(args.input, map_location="cpu")
    elif ext == ".npy":
        img = torch.from_numpy(np.load(args.input)).float()
    else:
        img = Image.open(args.input).convert("L")
        img = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
        
    if img.dim() == 2: img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3: img = img.unsqueeze(0)
    
    if img.shape[-2:] != (256, 256):
        img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
    
    img = img.to(device)
    
    # Run model
    predict_fn = load_model(args)
    with torch.no_grad():
        output = predict_fn(img)
        
    # Save visualization
    if args.target:
        target_ext = Path(args.target).suffix.lower()
        if target_ext == ".pt":
            target = torch.load(args.target, map_location="cpu")
        elif target_ext == ".npy":
            target = torch.from_numpy(np.load(args.target)).float()
        else:
            target = Image.open(args.target).convert("L")
            target = torch.from_numpy(np.array(target)).float() / 127.5 - 1.0
            
        if target.dim() == 2: target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 3: target = target.unsqueeze(0)
        if target.shape[-2:] != (256, 256):
            target = F.interpolate(target, size=(256, 256), mode='bilinear', align_corners=False)
        target = target.to(device)
        
        plot_translation_results(img, target, output, title=f"{args.architecture} {args.region}", save_path=args.output)
        
        # Calculate metrics
        calc = MetricsCalculator(device=args.device)
        metrics = calc.calculate_metrics(output, target)
        print(f"Metrics: {metrics}")
    else:
        # Just save output image
        out_img = (output[0, 0].cpu().numpy() + 1.0) / 2.0 * 255.0
        Image.fromarray(out_img.astype(np.uint8)).save(args.output)
        print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()
