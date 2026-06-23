#!/usr/bin/env python3
"""
Halton-MaskGIT Demo: Image → Tokens → Mask → Complete → Image
=============================================================

This script demonstrates the full pipeline:
  1. Download pretrained models (VQGAN + MaskGIT Transformer) from HuggingFace
  2. Load an input image
  3. Encode image to discrete tokens via VQGAN
  4. Simulate channel transmission (AWGN noise + quantization)
  5. Mask some tokens randomly
  6. Iteratively predict missing tokens using MaskGIT + Halton Sampler
  7. Decode tokens back to image
  8. Save a comparison image

Usage:
    python demo.py                          # uses a built-in example image
    python demo.py --image path/to/img.jpg  # your own image
    python demo.py --mask-ratio 0.30        # mask 30% of tokens
    python demo.py --steps 32               # sampling steps

Requirements:
    pip install torch torchvision pyyaml pillow huggingface_hub tqdm
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

# ── Add current dir to path for package imports ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def download_models():
    """Download pretrained VQGAN and MaskGIT weights from HuggingFace Hub."""
    saved_dir = Path("./saved_networks")
    saved_dir.mkdir(parents=True, exist_ok=True)

    vqgan_path = saved_dir / "vq_ds16_c2i.pt"
    vit_path = saved_dir / "ImageNet_384_large.pth"

    if not vqgan_path.exists():
        print("Downloading VQGAN weights (vq_ds16_c2i.pt)...")
        hf_hub_download(
            repo_id="FoundationVision/LlamaGen",
            filename="vq_ds16_c2i.pt",
            local_dir=str(saved_dir),
        )
    else:
        print("VQGAN weights already exist.")

    if not vit_path.exists():
        print("Downloading MaskGIT Transformer (ImageNet_384_large.pth)...")
        hf_hub_download(
            repo_id="llvictorll/Halton-Maskgit",
            filename="ImageNet_384_large.pth",
            local_dir=str(saved_dir),
        )
    else:
        print("MaskGIT Transformer weights already exist.")

    return str(vqgan_path), str(vit_path)


def build_model(cfg_path: str = "Config/base_cls2img.yaml"):
    """Load config, download weights, and initialize the MaskGIT model."""
    cfg = load_args_from_file(cfg_path)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.compile = False
    cfg.resume = True
    cfg.is_master = True
    cfg.is_multi_gpus = False
    cfg.global_rank = 0
    # Only use bfloat16 on CUDA-capable GPUs (Ampere or newer)
    if cfg.device == "cpu":
        cfg.dtype = "float32"

    # Download models if needed, then update paths
    vqgan_path, vit_path = download_models()
    cfg.vqgan_folder = vqgan_path
    cfg.vit_folder = vit_path
    cfg.vit_size = "large"
    cfg.img_size = 384

    print(f"Loading MaskGIT model on {cfg.device}...")
    model = MaskGIT(cfg)
    model.vit.eval()
    model.ae.eval()
    print("Model loaded successfully.")
    return cfg, model


def image_to_tensor(image_path: str, image_size: int, device: torch.device):
    """Load an image, resize, normalize to [-1, 1] and return as tensor."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(tensor: torch.Tensor):
    """Convert a [-1, 1] tensor to a PIL Image."""
    x = tensor.detach().cpu().squeeze(0)
    x = (x.clamp(-1, 1) + 1.0) / 2.0
    x = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(x)


def add_awgn(x: torch.Tensor, snr_db: float, generator=None):
    """Add AWGN noise to a tensor. Returns (noisy_tensor, noise_power)."""
    signal_power = torch.mean(x ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    noise = noise * torch.sqrt(noise_power)
    return x + noise, noise_power.item()


@torch.no_grad()
def encode_image(model, image_tensor, snr_db=None, generator=None):
    """
    Encode an image through VQGAN encoder → (optional AWGN) → quantize.
    Returns (quantized_latent, token_indices).
    """
    h = model.ae.encoder(image_tensor)
    h = model.ae.quant_conv(h)

    if snr_db is not None:
        h, noise_power = add_awgn(h, snr_db, generator=generator)

    quant, _, info = model.ae.quantize(h)
    token_indices = info[2]  # shape: (b, h*w)

    h_size = image_tensor.shape[-2] // model.args.f_factor
    w_size = image_tensor.shape[-1] // model.args.f_factor
    tokens = token_indices.view(image_tensor.shape[0], h_size, w_size)
    return quant, tokens


@torch.no_grad()
def decode_tokens(model, tokens):
    """Decode token indices back to an image through VQGAN decoder."""
    img = model.ae.decode_code(tokens)
    return torch.clamp(img, -1, 1)


def compute_psnr(x: torch.Tensor, y: torch.Tensor):
    """Compute PSNR between two tensors in [-1, 1] range."""
    mse = torch.mean((x - y) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * torch.log10(4.0 / mse))


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    image_path: str,
    mask_ratio: float = 0.30,
    steps: int = 32,
    label: int = 1,
    seed: int = 42,
    output_dir: str = "outputs",
    with_noise: bool = True,
    snr_db: float = 10.0,
):
    """
    Run the full image → tokens → mask → complete → image pipeline.

    Args:
        image_path: Path to input image
        mask_ratio: Fraction of tokens to mask (0.0 to 1.0)
        steps: Number of iterative sampling steps
        label: ImageNet class label for conditional generation
        seed: Random seed
        output_dir: Where to save results
        with_noise: Add AWGN noise to simulate channel transmission
        snr_db: SNR in dB for the AWGN noise
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── 1. Build model ─────────────────────────────────────────────────
    cfg, model = build_model()

    # ── 2. Create sampler ──────────────────────────────────────────────
    sampler = HaltonSampler(
        sm_temp_min=1.0,
        sm_temp_max=1.2,
        temp_pow=1.0,
        temp_warmup=0,
        w=2.0,            # classifier-free guidance weight
        sched_pow=2.0,
        step=steps,
        randomize=False,
        top_k=-1,
    )
    print(f"Sampler: {sampler}")

    # ── 3. Load & encode image ─────────────────────────────────────────
    print(f"\nLoading image: {image_path}")
    img_tensor = image_to_tensor(image_path, cfg.img_size, cfg.device)
    print(f"Image shape: {img_tensor.shape}")

    # Encode without noise to get clean tokens
    print("Encoding image to tokens...")
    _, clean_tokens = encode_image(model, img_tensor, snr_db=None)

    # Encode with noise (simulating channel transmission)
    if with_noise:
        print(f"Encoding with AWGN noise (SNR={snr_db}dB)...")
        generator = torch.Generator(device=cfg.device).manual_seed(seed)
        _, noisy_tokens = encode_image(model, img_tensor, snr_db=snr_db, generator=generator)
    else:
        noisy_tokens = clean_tokens.clone()

    # Reconstruct VQ image (no mask, no completion)
    vq_recon = decode_tokens(model, noisy_tokens)
    vq_psnr = compute_psnr(img_tensor, vq_recon)
    print(f"VQ reconstruction PSNR: {vq_psnr:.2f} dB")

    # ── 4. Create random mask ──────────────────────────────────────────
    b, h, w = noisy_tokens.shape
    n_tokens = h * w
    k_mask = max(1, int(round(n_tokens * mask_ratio)))
    print(f"\nMasking {k_mask}/{n_tokens} tokens ({mask_ratio:.0%})")

    mask = torch.zeros((b, h, w), dtype=torch.bool, device=cfg.device)
    perm = torch.randperm(n_tokens, device=cfg.device)
    mask.view(-1)[perm[:k_mask]] = True

    # Create masked token tensor
    masked_tokens = noisy_tokens.clone()
    masked_tokens[mask] = cfg.mask_value

    # Generate a visual preview: VQ image with masked regions blacked out
    mask_vis = F.interpolate(
        mask.float().unsqueeze(0),  # (1, 1, h, w)
        size=(cfg.img_size, cfg.img_size),
        mode="nearest",
    ).squeeze(0).squeeze(0)  # (H, W)
    vq_np = (tensor_to_pil(vq_recon))
    vq_arr = np.array(vq_np).copy()
    vq_arr[mask_vis.cpu().numpy() > 0.5] = [0, 0, 0]
    masked_preview = Image.fromarray(vq_arr)

    # ── 5. Run MaskGIT completion ──────────────────────────────────────
    print(f"\nRunning MaskGIT completion ({steps} steps)...")
    labels = torch.LongTensor([label]).to(cfg.device)

    gen_images, _, _, predicted_tokens = sampler(
        trainer=model,
        init_code=masked_tokens,
        nb_sample=1,
        labels=labels,
        verbose=True,
        edit_mask=mask,
        return_final_code=True,
    )

    # Count correctly predicted tokens
    predicted_tokens_corrected = predicted_tokens.clone()
    predicted_tokens_corrected[~mask] = noisy_tokens[~mask]
    token_acc = (predicted_tokens_corrected == noisy_tokens).float().mean().item()
    print(f"Token prediction accuracy (all tokens): {token_acc:.1%}")

    # ── 6. Results ─────────────────────────────────────────────────────
    gen_psnr = compute_psnr(img_tensor, gen_images)
    print(f"Completed image PSNR: {gen_psnr:.2f} dB")

    # Compute PSNR gain from completion
    masked_only_recon = decode_tokens(model, masked_tokens)
    masked_psnr = compute_psnr(img_tensor, masked_only_recon)
    print(f"Masked-only PSNR: {masked_psnr:.2f} dB")

    # ── 7. Save results ────────────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_stem = Path(image_path).stem
    orig_pil = tensor_to_pil(img_tensor)
    vq_pil = tensor_to_pil(vq_recon)
    gen_pil = tensor_to_pil(gen_images)

    # Save individual images
    orig_pil.save(out_path / f"{img_stem}_1_original.png")
    vq_pil.save(out_path / f"{img_stem}_2_vq_recon_{vq_psnr:.1f}dB.png")
    masked_preview.save(out_path / f"{img_stem}_3_masked_{mask_ratio:.0%}.png")
    gen_pil.save(out_path / f"{img_stem}_4_completed_{gen_psnr:.1f}dB.png")

    # Create comparison grid
    w_img = orig_pil.width
    h_img = orig_pil.height
    gap = 8
    canvas_w = w_img * 4 + gap * 5
    canvas_h = h_img + gap * 2 + 40

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)

    titles = [
        "Original",
        f"VQ Recon ({vq_psnr:.1f}dB)",
        f"Masked {mask_ratio:.0%}",
        f"Completed ({gen_psnr:.1f}dB)",
    ]
    panels = [orig_pil, vq_pil, masked_preview, gen_pil]

    for i, (title, panel) in enumerate(zip(titles, panels)):
        x = gap + i * (w_img + gap)
        y_top = 36
        draw.text((x, 10), title, fill=(230, 230, 230))
        canvas.paste(panel, (x, y_top))

    comparison_path = out_path / f"{img_stem}_comparison.png"
    canvas.save(comparison_path)
    print(f"\nSaved results to: {out_path.resolve()}/")
    print(f"  {img_stem}_1_original.png")
    print(f"  {img_stem}_2_vq_recon_{vq_psnr:.1f}dB.png")
    print(f"  {img_stem}_3_masked_{mask_ratio:.0%}.png")
    print(f"  {img_stem}_4_completed_{gen_psnr:.1f}dB.png")
    print(f"  {img_stem}_comparison.png  ← all panels")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Image:                {image_path}")
    print(f"  Device:               {cfg.device}")
    print(f"  Mask ratio:           {mask_ratio:.0%} ({k_mask}/{n_tokens} tokens)")
    print(f"  Sampling steps:       {steps}")
    print(f"  Channel AWGN:         {'Yes' if with_noise else 'No'}" +
          (f", SNR={snr_db}dB" if with_noise else ""))
    print(f"  VQ recon PSNR:        {vq_psnr:.2f} dB")
    print(f"  Masked-only PSNR:     {masked_psnr:.2f} dB")
    print(f"  Completed PSNR:       {gen_psnr:.2f} dB")
    print(f"  Token accuracy:       {token_acc:.1%}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Halton-MaskGIT: Image → Tokens → Mask → Complete → Image"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to input image. If not provided, uses a built-in demo."
    )
    parser.add_argument(
        "--mask-ratio", type=float, default=0.30,
        help="Fraction of tokens to mask (default: 0.30)"
    )
    parser.add_argument(
        "--steps", type=int, default=32,
        help="Number of iterative sampling steps (default: 32)"
    )
    parser.add_argument(
        "--label", type=int, default=1,
        help="ImageNet class label for conditioning (default: 1 = goldfish)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--no-noise", action="store_true",
        help="Disable channel AWGN noise (pure VQ)"
    )
    parser.add_argument(
        "--snr", type=float, default=10.0,
        help="SNR in dB for channel AWGN noise (default: 10.0)"
    )
    args = parser.parse_args()

    # If no image provided, use a built-in example
    if args.image is None:
        print("No image provided. Trying to use a demo image...")
        # Check for images in common locations
        candidates = [
            Path("../Halton-MaskGIT-main/data/top200_psnr_images/001_000000003264_psnr_29.238.jpg"),
            Path("data/top200_psnr_images/001_000000003264_psnr_29.238.jpg"),
        ]
        found = False
        for cand in candidates:
            if cand.exists():
                args.image = str(cand)
                found = True
                print(f"Using: {args.image}")
                break
        if not found:
            print("No demo image found. Please provide an image with --image")
            print("Example: python demo.py --image path/to/your/image.jpg")
            sys.exit(1)

    run_pipeline(
        image_path=args.image,
        mask_ratio=args.mask_ratio,
        steps=args.steps,
        label=args.label,
        seed=args.seed,
        output_dir=args.output,
        with_noise=not args.no_noise,
        snr_db=args.snr,
    )


if __name__ == "__main__":
    main()
