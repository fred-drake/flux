# Containerized Flux for 24gb GPUs

Code repo for article: [Containerized Flux](https://codingwithcody.com/2025/03/09/containerized-flux/)

## Usage

Run Flux image generation with the following command line arguments:

```bash
python main.py --prompt "your text prompt here" [options]
```

### Arguments

- `--prompt` (required): Text prompt for image generation
- `--width` (default: 512): Image width in pixels
- `--height` (default: 512): Image height in pixels
- `--steps` (default: 20): Number of inference steps
- `--seed` (default: 42): Random seed for reproducibility
- `--randgen X`: Generate X images with random seeds (ignores --seed)
- `--output`: Output file path (default: /app/output/picture-TIMESTAMP.png)
- `--model`: Model to use: `dev` (default) or `schnell`
- `--meta`: Save metadata about this run to /app/output/meta-TIMESTAMP.json

### Examples

Generate a single image:
```bash
python main.py --prompt "a beautiful sunset over mountains" --width 1024 --height 768
```

Generate multiple images with random seeds:
```bash
python main.py --prompt "abstract art" --randgen 5 --steps 30
```

Use the schnell model with metadata:
```bash
python main.py --prompt "futuristic city" --model schnell --meta
```
