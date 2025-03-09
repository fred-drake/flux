import argparse
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel
from PIL import Image

# Use GPU if available, else fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Loads the Flux Dev model in 8-bit mode for low GPU memory."""
    print("Loading model...")

    text_encoder_8bit = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="text_encoder_2",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
    )

    transformer_8bit = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
    )

    return FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
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
    """Runs the Flux Dev inference using CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Flux Dev image generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="output/generated_image.png", help="Output file path")

    args = parser.parse_args()

    model = load_model()
    image = generate_image(model, args.prompt, args.width, args.height, args.steps, args.seed)

    image.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()
