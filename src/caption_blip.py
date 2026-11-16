import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_ID = "Salesforce/blip-image-captioning-base"


def generate_caption(model, processor, image_path, device):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=30
        )

    return processor.decode(
        output[0],
        skip_special_tokens=True
    )


def main(image_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    image_dir = Path(image_dir)

    image_files = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    results = []

    for image_path in image_files:
        print(f"Processing: {image_path.name}")

        caption = generate_caption(
            model,
            processor,
            image_path,
            device
        )

        print(f"  -> {caption}")

        results.append({
            "image": image_path.name,
            "caption": caption
        })

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "blip_captions.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/caption_blip.py <image_directory>")
        raise SystemExit(1)

    main(sys.argv[1])