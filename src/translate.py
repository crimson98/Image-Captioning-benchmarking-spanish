import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_ID = "Helsinki-NLP/opus-mt-en-es"


def translate_text(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=64
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


def main(input_file):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Loading translator: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    input_path = Path(input_file)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    translation_start = time.perf_counter()
    for item in data:
        caption_en = item["caption_en"]

        caption_es = translate_text(
            caption_en,
            tokenizer,
            model,
            device
        )

        print(f"\nEN: {caption_en}")
        print(f"ES: {caption_es}")

        results.append({
            **item,
            "caption_es": caption_es
        })

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    translation_end = time.perf_counter()

    translation_seconds = (
        translation_end - translation_start
    )

    seconds_per_image = (
        translation_seconds / len(results)
    )

    peak_vram_gb = None

    if torch.cuda.is_available():
        peak_vram_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )
    
    output_path = Path("results/xm3600_blip_es.json")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    metrics = {
        "num_images": len(results),
        "translation_seconds": translation_seconds,
        "seconds_per_image": seconds_per_image,
        "peak_vram_gb": peak_vram_gb
    }

    with open(
        "results/translation_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python src/translate.py "
            "results/xm3600_blip.json"
        )
        raise SystemExit(1)

    main(sys.argv[1])