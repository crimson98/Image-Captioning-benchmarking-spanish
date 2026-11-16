import json
import os
import platform
import shutil
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path

import torch
import transformers


CAPTIONS_FILE = "data/xm3600/captions/captions.jsonl"
IMAGES_DIR = "data/xm3600/images"


def run_command(command):
    start = time.perf_counter()

    subprocess.run(
        command,
        check=True
    )

    return time.perf_counter() - start


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )

        return result.stdout.strip()

    except Exception:
        return None


def get_system_info():
    gpu = None

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)

        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_gb": round(
                properties.total_memory / (1024 ** 3),
                2
            )
        }

    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": (
            platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER")
            or "unknown"
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu
    }


def main(limit):
    run_started = datetime.now()
    run_id = run_started.strftime("%Y%m%d_%H%M%S")

    run_dir = Path(
        f"results/runs/{run_id}_n{limit}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    python = sys.executable

    timings = {}

    timings["load_dataset_seconds"] = run_command([
        python,
        "src/load_xm3600.py",
        CAPTIONS_FILE,
        IMAGES_DIR,
        str(limit)
    ])

    timings["blip_total_seconds"] = run_command([
        python,
        "src/caption_xm3600.py",
        "results/xm3600_sample.json"
    ])

    timings["translation_total_seconds"] = run_command([
        python,
        "src/translate.py",
        "results/xm3600_blip.json"
    ])

    timings["cider_seconds"] = run_command([
        python,
        "src/evaluate_cider.py",
        "results/xm3600_blip_es.json"
    ])

    with open(
        "results/xm3600_cider.json",
        "r",
        encoding="utf-8"
    ) as f:
        cider_result = json.load(f)

    with open(
        "results/blip_metrics.json",
        "r",
        encoding="utf-8"
    ) as f:
        blip_metrics = json.load(f)

    with open(
        "results/translation_metrics.json",
        "r",
        encoding="utf-8"
    ) as f:
        translation_metrics = json.load(f)

    summary = {
        "run_id": run_id,
        "timestamp": run_started.isoformat(),
        "num_images": limit,

        "models": {
            "caption_model":
                "Salesforce/blip-image-captioning-base",
            "translation_model":
                "Helsinki-NLP/opus-mt-en-es"
        },

        "system": get_system_info(),

        "timings": timings,

        "blip": blip_metrics,

        "translation": translation_metrics,

        "metrics": {
            "cider_mean":
                cider_result["cider_mean"]
        },

        "git_commit": get_git_commit()
    }

    with open(
        run_dir / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    files = [
        "results/xm3600_sample.json",
        "results/xm3600_blip.json",
        "results/xm3600_blip_es.json",
        "results/xm3600_cider.json",
        "results/blip_metrics.json",
        "results/translation_metrics.json"
    ]

    for filename in files:
        path = Path(filename)

        if path.exists():
            shutil.copy2(
                path,
                run_dir / path.name
            )

    print("\n==============================")
    print("PoC finished")
    print("==============================")

    print(f"Images: {limit}")

    print(
        f"CIDEr mean: "
        f"{cider_result['cider_mean']:.4f}"
    )

    print(
        f"BLIP inference: "
        f"{blip_metrics['inference_seconds']:.2f}s"
    )

    print(
        f"BLIP/image: "
        f"{blip_metrics['seconds_per_image']:.3f}s"
    )

    print(
        f"BLIP peak VRAM: "
        f"{blip_metrics['peak_vram_gb']}"
    )

    print(
        f"Translation/image: "
        f"{translation_metrics['seconds_per_image']:.3f}s"
    )

    print(f"\nSaved in: {run_dir}")


if __name__ == "__main__":
    limit = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else 100
    )

    main(limit)