import argparse
import torch
import json
import random
from datetime import datetime
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel
from PIL import Image

# Use GPU if available, else fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name: str = "dev"):
    """Loads the specified Flux model in 8-bit mode for low GPU memory."""
    print(f"Loading {model_name} model...")
    
    if model_name == "schnell":
        model_id = "black-forest-labs/FLUX.1-schnell"
    else:  # default to dev
        model_id = "black-forest-labs/FLUX.1-dev"

    text_encoder_8bit = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
    )

    transformer_8bit = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
    )

    return FluxPipeline.from_pretrained(
        model_id,
        text_encoder_2=text_encoder_8bit,
        transformer=transformer_8bit,
        torch_dtype=torch.float16,
        device_map="balanced",
    )

@torch.no_grad()
def generate_image(pipe, prompt: str, width: int, height: int, steps: int, seed: int):
    """Generates an image from a text prompt using Flux Dev."""
    print(f"Generating image for: '{prompt}'")

    generator = torch.Generator(device).manual_seed(seed)
    images = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=steps,
        num_images_per_prompt=1,
        generator=generator,
        width=width,
        height=height,
    ).images

    return images[0]

def main():
    """Runs the Flux inference using CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Flux image generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--randgen", type=int, help="Generate X images with random seeds (ignores --seed)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--model", type=str, default="dev", choices=["dev", "schnell"], help="Model to use: dev (default) or schnell")
    parser.add_argument("--meta", action="store_true", help="Save metadata about this run to /app/output/meta-TIMESTAMP.json")

    args = parser.parse_args()
    
    # Generate timestamp once for both files
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    model = load_model(args.model)
    
    # Handle random generation or single image
    if args.randgen:
        num_images = args.randgen
        for i in range(num_images):
            seed = random.randint(0, 2147483647)  # int32 upper bound
            
            # Set output path for each image
            if args.output is None:
                output_path = f"/app/output/picture-{timestamp}-{i+1:03d}.png"
            else:
                # Insert sequence number before file extension
                name, ext = args.output.rsplit('.', 1) if '.' in args.output else (args.output, 'png')
                output_path = f"{name}-{i+1:03d}.{ext}"
            
            image = generate_image(model, args.prompt, args.width, args.height, args.steps, seed)
            image.save(output_path)
            print(f"Image {i+1}/{num_images} saved to {output_path}")
            
            # Save metadata if requested
            if args.meta:
                meta_path = f"/app/output/meta-{timestamp}-{i+1:03d}.json"
                metadata = {
                    "model": args.model,
                    "prompt": args.prompt,
                    "height": args.height,
                    "width": args.width,
                    "steps": args.steps,
                    "seed": seed,
                    "output": output_path
                }
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Metadata {i+1}/{num_images} saved to {meta_path}")
    else:
        seed = args.seed
        
        # Set default output path if not provided
        if args.output is None:
            args.output = f"/app/output/picture-{timestamp}.png"

        image = generate_image(model, args.prompt, args.width, args.height, args.steps, seed)
        image.save(args.output)
        print(f"Image saved to {args.output}")
        
        # Save metadata if requested
        if args.meta:
            meta_path = f"/app/output/meta-{timestamp}.json"
            metadata = {
                "model": args.model,
                "prompt": args.prompt,
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "seed": seed,
                "output": args.output
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()
