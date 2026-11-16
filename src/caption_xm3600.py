import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_ID = "Salesforce/blip-image-captioning-base"


def main(input_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    input_path = Path(input_file)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

        inference_start = time.perf_counter()
    
    for item in data:
        image_path = Path(item["image_path"])

        print(f"\nProcessing: {image_path}")

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=50
            )

        caption_en = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        print(f"EN: {caption_en}")

        results.append({
            **item,
            "caption_en": caption_en
        })

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_end = time.perf_counter()

    inference_seconds = inference_end - inference_start
    seconds_per_image = inference_seconds / len(results)

    peak_vram_gb = None

    if torch.cuda.is_available():
        peak_vram_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print(f"\nBLIP inference time: {inference_seconds:.2f}s")
    print(f"BLIP time/image: {seconds_per_image:.3f}s")

    if peak_vram_gb is not None:
        print(f"BLIP peak VRAM: {peak_vram_gb:.3f} GB")

    output_path = Path("results/xm3600_blip.json")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    metrics = {
        "num_images": len(results),
        "inference_seconds": inference_seconds,
        "seconds_per_image": seconds_per_image,
        "peak_vram_gb": peak_vram_gb
    }

    with open(
        "results/blip_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---- mostrar resumen ----

    print(f"\nSaved to: {output_path}")
    print(f"BLIP inference time: {inference_seconds:.2f}s")
    print(f"BLIP time/image: {seconds_per_image:.3f}s")

    if peak_vram_gb is not None:
        print(f"BLIP peak VRAM: {peak_vram_gb:.3f} GB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python src/caption_xm3600.py "
            "results/xm3600_sample.json"
        )
        raise SystemExit(1)

    main(sys.argv[1])