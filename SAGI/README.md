# Halton-MaskGIT

Official implementation of Halton-MaskGIT: Masked Generative Image Transformer with Halton Sequence Sampling.

## Installation

```bash
pip install torch torchvision pyyaml pillow huggingface_hub tqdm
```

## Usage

### Command Line

```bash
# Basic usage (pretrained weights downloaded automatically on first run)
python demo.py --image path/to/image.jpg

# Custom mask ratio and sampling steps
python demo.py --image cat.jpg --mask-ratio 0.30 --steps 32

# Disable channel noise
python demo.py --image cat.jpg --no-noise
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image` | `str` | *required* | Path to input image |
| `--mask-ratio` | `float` | `0.30` | Fraction of tokens to mask |
| `--steps` | `int` | `32` | Number of iterative decoding steps |
| `--label` | `int` | `1` | ImageNet class label for conditioning |
| `--seed` | `int` | `42` | Random seed |
| `--output` | `str` | `outputs` | Output directory |
| `--no-noise` | `flag` | `False` | Disable AWGN channel noise |
| `--snr` | `float` | `10.0` | Channel SNR in dB |

### Python API

```python
from Sampler.halton_sampler import HaltonSampler
from Trainer.cls_trainer import MaskGIT
from Utils.utils import load_args_from_file

cfg = load_args_from_file("Config/base_cls2img.yaml")
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
model = MaskGIT(cfg)

sampler = HaltonSampler(
    sm_temp_min=1.0, sm_temp_max=1.2, temp_pow=1.0, temp_warmup=0,
    w=2.0, sched_pow=2.0, step=32, randomize=False, top_k=-1,
)

labels = torch.LongTensor([1]).to(cfg.device)
images, codes, masks = sampler(trainer=model, nb_sample=1, labels=labels)
```

## Pretrained Models

Weights are downloaded automatically from HuggingFace Hub:

| Model | Source |
|-------|--------|
| VQGAN (VQ-16) | [`FoundationVision/LlamaGen`](https://huggingface.co/FoundationVision/LlamaGen) |
| MaskGIT Transformer (Large, 384²) | [`llvictorll/Halton-Maskgit`](https://huggingface.co/llvictorll/Halton-Maskgit) |

## License

MIT License. VQGAN adapted from [LlamaGen](https://github.com/FoundationVision/LlamaGen). Transformer follows the [DiT](https://github.com/facebookresearch/DiT) architecture.
